from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from acd_bidask_forward import NY, build_atr, metrics, resample_5m
from acd_bidask_from_csv_v2 import read_side

DATA_DIR = Path("data_bidask")
OUT = Path("artifacts_2b")
OUT.mkdir(exist_ok=True)
START = pd.Timestamp("2024-01-01").date()
END = pd.Timestamp("2026-07-16").date()


@dataclass(frozen=True)
class Config:
    reference: str
    sweep_atr: float
    confirm_mode: str
    rr: float
    entry_cutoff: str
    side_mode: str
    trend_filter: bool
    stop_buffer_atr: float
    extra_cost: float = 0.05


@dataclass
class Trade:
    date: str
    year: int
    reference: str
    side: str
    entry_time: str
    exit_time: str
    entry: float
    exit: float
    stop: float
    target: float
    risk: float
    net_points: float
    r_net: float
    exit_reason: str
    entry_spread: float


@dataclass
class DayContext:
    local_day: object
    year: int
    day: pd.DataFrame
    daily_atr: float
    references: dict[str, tuple[float, float]]
    pos_1030: np.ndarray
    pos_1200: np.ndarray


def positions(index: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    sh, sm = map(int, start.split(":"))
    eh, em = map(int, end.split(":"))
    minute = index.hour * 60 + index.minute
    return np.flatnonzero((minute >= sh * 60 + sm) & (minute < eh * 60 + em))


def prepare_contexts(bars: pd.DataFrame, atr: pd.Series) -> list[DayContext]:
    groups = {day: frame.sort_index() for day, frame in bars.groupby(bars.index.date, sort=True)}
    trading_days = sorted(groups)
    prior_map = {trading_days[i]: trading_days[i - 1] for i in range(1, len(trading_days))}
    atr_dates = np.array(list(atr.index), dtype=object)
    atr_values = atr.to_numpy(dtype=float)
    contexts: list[DayContext] = []

    for local_day in trading_days:
        if local_day < START or local_day > END or pd.Timestamp(local_day).weekday() >= 5:
            continue
        day = groups[local_day]
        utc_day = day.index[0].tz_convert("UTC").date()
        atr_idx = int(np.searchsorted(atr_dates, utc_day, side="left")) - 1
        if atr_idx < 0:
            continue
        daily_atr = float(atr_values[atr_idx])
        if not np.isfinite(daily_atr) or daily_atr <= 0:
            continue

        day_start = pd.Timestamp(local_day, tz=NY)
        refs: dict[str, tuple[float, float]] = {}
        pre = day.loc[day_start + pd.Timedelta(hours=8, minutes=30): day_start + pd.Timedelta(hours=9, minutes=29)]
        if len(pre) >= 3:
            refs["premarket"] = (float(pre.mid_high.max()), float(pre.mid_low.min()))
        overnight = bars.loc[day_start - pd.Timedelta(hours=6): day_start + pd.Timedelta(hours=9, minutes=29)]
        if len(overnight) >= 3:
            refs["overnight"] = (float(overnight.mid_high.max()), float(overnight.mid_low.min()))
        prior_day = prior_map.get(local_day)
        if prior_day is not None:
            prior = groups[prior_day]
            if len(prior) >= 3:
                refs["prior_day"] = (float(prior.mid_high.max()), float(prior.mid_low.min()))
        if not refs:
            continue

        contexts.append(
            DayContext(
                local_day=local_day,
                year=int(str(local_day)[:4]),
                day=day,
                daily_atr=daily_atr,
                references=refs,
                pos_1030=positions(day.index, "09:30", "10:30"),
                pos_1200=positions(day.index, "09:30", "12:00"),
            )
        )
    return contexts


def find_signal(ctx: DayContext, cfg: Config):
    if cfg.reference not in ctx.references:
        return None
    ref_high, ref_low = ctx.references[cfg.reference]
    active = ctx.pos_1030 if cfg.entry_cutoff == "10:30" else ctx.pos_1200
    if len(active) < 2:
        return None
    high_trigger = ref_high + ctx.daily_atr * cfg.sweep_atr
    low_trigger = ref_low - ctx.daily_atr * cfg.sweep_atr
    max_wait = 0 if cfg.confirm_mode == "same_bar" else 3
    candidates = []

    if cfg.side_mode != "long_only":
        for pos_raw in active:
            pos = int(pos_raw)
            bar = ctx.day.iloc[pos]
            if bar.ask_high < high_trigger:
                continue
            trend_ok = (not cfg.trend_filter) or (
                pos > 0 and float(ctx.day.iloc[pos - 1].mid_close) > float(ctx.day.iloc[pos - 1].ema200)
            )
            if trend_ok:
                for wait in range(max_wait + 1):
                    confirm_pos = pos + wait
                    if confirm_pos >= len(ctx.day) or confirm_pos > int(active[-1]):
                        break
                    if float(ctx.day.iloc[confirm_pos].mid_close) < ref_high:
                        entry_pos = confirm_pos + 1
                        if entry_pos < len(ctx.day) and entry_pos <= int(active[-1]):
                            candidates.append((entry_pos, -1, float(bar.ask_high)))
                        break
            break

    if cfg.side_mode != "short_only":
        for pos_raw in active:
            pos = int(pos_raw)
            bar = ctx.day.iloc[pos]
            if bar.bid_low > low_trigger:
                continue
            trend_ok = (not cfg.trend_filter) or (
                pos > 0 and float(ctx.day.iloc[pos - 1].mid_close) < float(ctx.day.iloc[pos - 1].ema200)
            )
            if trend_ok:
                for wait in range(max_wait + 1):
                    confirm_pos = pos + wait
                    if confirm_pos >= len(ctx.day) or confirm_pos > int(active[-1]):
                        break
                    if float(ctx.day.iloc[confirm_pos].mid_close) > ref_low:
                        entry_pos = confirm_pos + 1
                        if entry_pos < len(ctx.day) and entry_pos <= int(active[-1]):
                            candidates.append((entry_pos, 1, float(bar.bid_low)))
                        break
            break

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None
    return candidates[0]


def simulate(ctx: DayContext, cfg: Config, signal):
    entry_pos, direction, extreme = signal
    entry_bar = ctx.day.iloc[entry_pos]
    stop_buffer = ctx.daily_atr * cfg.stop_buffer_atr
    if direction > 0:
        entry = float(entry_bar.ask_open)
        stop = extreme - stop_buffer
        risk = entry - stop
        target = entry + risk * cfg.rr
    else:
        entry = float(entry_bar.bid_open)
        stop = extreme + stop_buffer
        risk = stop - entry
        target = entry - risk * cfg.rr
    if not np.isfinite(risk) or risk <= 0:
        return None

    after = ctx.day.index[entry_pos:]
    noon = np.flatnonzero((after.hour * 60 + after.minute) >= 12 * 60)
    last_pos = entry_pos + int(noon[0]) if len(noon) else len(ctx.day) - 1
    exit_price = float(ctx.day.iloc[last_pos].bid_open if direction > 0 else ctx.day.iloc[last_pos].ask_open)
    reason = "TIME"

    for pos in range(entry_pos, last_pos + 1):
        bar = ctx.day.iloc[pos]
        if direction > 0:
            if bar.bid_low <= stop:
                exit_price = min(stop, float(bar.bid_open))
                last_pos = pos
                reason = "STOP"
                break
            if bar.bid_high >= target:
                exit_price = target
                last_pos = pos
                reason = "TP"
                break
        else:
            if bar.ask_high >= stop:
                exit_price = max(stop, float(bar.ask_open))
                last_pos = pos
                reason = "STOP"
                break
            if bar.ask_low <= target:
                exit_price = target
                last_pos = pos
                reason = "TP"
                break

    net = direction * (exit_price - entry) - cfg.extra_cost
    return Trade(
        date=str(ctx.local_day),
        year=ctx.year,
        reference=cfg.reference,
        side="LONG" if direction > 0 else "SHORT",
        entry_time=ctx.day.index[entry_pos].isoformat(),
        exit_time=ctx.day.index[last_pos].isoformat(),
        entry=entry,
        exit=exit_price,
        stop=stop,
        target=target,
        risk=risk,
        net_points=net,
        r_net=net / risk,
        exit_reason=reason,
        entry_spread=float(entry_bar.ask_open - entry_bar.bid_open),
    )


def backtest(contexts: list[DayContext], cfg: Config):
    trades = []
    for ctx in contexts:
        signal = find_signal(ctx, cfg)
        if signal is None:
            continue
        trade = simulate(ctx, cfg, signal)
        if trade is not None:
            trades.append(trade)
    return trades


def period_metrics(trades, year):
    return metrics([trade for trade in trades if trade.year == year])


def main():
    bid = read_side(DATA_DIR / "xauusd_bid_m1.csv", "bid")
    ask = read_side(DATA_DIR / "xauusd_ask_m1.csv", "ask")
    m1 = bid.join(ask, how="inner")
    bars_utc = resample_5m(m1)
    atr = build_atr(bars_utc)
    contexts = prepare_contexts(bars_utc.tz_convert(NY), atr)

    configs = [
        Config(reference, sweep_atr, confirm, rr, cutoff, side, trend, stop_buffer)
        for reference, sweep_atr, confirm, rr, cutoff, side, trend, stop_buffer in itertools.product(
            ["prior_day", "overnight", "premarket"],
            [0.00, 0.02, 0.05],
            ["same_bar", "within_3"],
            [1.0, 1.5, 2.0],
            ["10:30", "12:00"],
            ["both", "long_only", "short_only"],
            [False, True],
            [0.00, 0.01],
        )
    ]

    rows = []
    cache = {}
    for i, cfg in enumerate(configs, 1):
        trades = backtest(contexts, cfg)
        cache[cfg] = trades
        row = asdict(cfg)
        row.update({f"all_{key}": value for key, value in metrics(trades).items()})
        for year in [2024, 2025, 2026]:
            row.update({f"y{year}_{key}": value for key, value in period_metrics(trades, year).items()})
        rows.append(row)
        if i % 150 == 0:
            print(f"processed {i}/{len(configs)}")

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "two_b_grid.csv", index=False)
    discovery = results[
        (results.y2024_trades >= 25)
        & np.isfinite(results.y2024_profit_factor)
        & (results.y2024_profit_factor > 1.0)
        & (results.y2024_avg_r > 0.0)
    ].copy()
    discovery["score"] = discovery.y2024_avg_r.clip(-1, 2) * np.sqrt(discovery.y2024_trades) - 0.03 * discovery.y2024_max_dd_r
    discovery = discovery.sort_values(["score", "y2024_profit_factor"], ascending=False)
    discovery.to_csv(OUT / "two_b_discovery_ranked.csv", index=False)

    robust = discovery[
        (discovery.y2025_trades >= 20)
        & (discovery.y2026_trades >= 8)
        & (discovery.y2025_profit_factor > 1.0)
        & (discovery.y2026_profit_factor > 1.0)
        & (discovery.y2025_avg_r > 0.0)
        & (discovery.y2026_avg_r > 0.0)
    ].copy()
    robust.to_csv(OUT / "two_b_forward_positive.csv", index=False)
    strict = robust[
        (robust.all_profit_factor >= 1.20)
        & (robust.all_avg_r >= 0.08)
        & (robust.all_max_dd_r <= 10.0)
    ].copy()
    strict.to_csv(OUT / "two_b_strict_pass.csv", index=False)

    if not discovery.empty:
        best = discovery.iloc[0]
        best_cfg = Config(
            str(best.reference), float(best.sweep_atr), str(best.confirm_mode), float(best.rr),
            str(best.entry_cutoff), str(best.side_mode), bool(best.trend_filter), float(best.stop_buffer_atr),
            float(best.extra_cost),
        )
        pd.DataFrame([asdict(t) for t in cache[best_cfg]]).to_csv(OUT / "best_discovery_trades.csv", index=False)

    columns = [
        "reference", "sweep_atr", "confirm_mode", "rr", "entry_cutoff", "side_mode", "trend_filter", "stop_buffer_atr",
        "y2024_trades", "y2024_win_rate", "y2024_profit_factor", "y2024_avg_r", "y2024_max_dd_r",
        "y2025_trades", "y2025_win_rate", "y2025_profit_factor", "y2025_avg_r", "y2025_max_dd_r",
        "y2026_trades", "y2026_win_rate", "y2026_profit_factor", "y2026_avg_r", "y2026_max_dd_r",
        "all_trades", "all_profit_factor", "all_avg_r", "all_max_dd_r",
    ]
    report = [
        "# NY Open 2B / Liquidity-Sweep True BID/ASK Test",
        "",
        "Research adaptation of Sperandeo 2B: price exceeds a prior high/low, fails to carry through, and closes back through the old level.",
        "Data: Dukascopy BID/ASK M1 resampled to M5; ASK/BID-specific execution; extra cost 0.05; same-bar stop/target ambiguity counted as stop.",
        "Selection: 2024 only. Validation: 2025. Forward test: 2026 through July 16.",
        "",
        "## Top 15 selected only on 2024",
        "",
        discovery.head(15)[columns].to_markdown(index=False, floatfmt=".3f") if not discovery.empty else "No discovery candidate.",
        "",
        "## Positive in both 2025 and 2026 YTD",
        "",
        f"Count: {len(robust)}",
        "",
        robust.head(30)[columns].to_markdown(index=False, floatfmt=".3f") if not robust.empty else "None passed.",
        "",
        "## Strict target passed",
        "",
        f"Count: {len(strict)}",
        "",
        strict.head(30)[columns].to_markdown(index=False, floatfmt=".3f") if not strict.empty else "None passed.",
    ]
    (OUT / "two_b_report.md").write_text("\n".join(report), encoding="utf-8")
    print((OUT / "two_b_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
