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
    sweep_size_atr: float
    entry_spread: float


def positions(index: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    sh, sm = map(int, start.split(":"))
    eh, em = map(int, end.split(":"))
    minute = index.hour * 60 + index.minute
    return np.flatnonzero((minute >= sh * 60 + sm) & (minute < eh * 60 + em))


def prior_atr(atr: pd.Series, utc_day) -> float | None:
    values = atr.loc[atr.index < utc_day]
    return None if values.empty else float(values.iloc[-1])


def reference_levels(bars: pd.DataFrame, local_day, reference: str):
    day_start = pd.Timestamp(local_day, tz=NY)
    if reference == "premarket":
        segment = bars.loc[day_start + pd.Timedelta(hours=8, minutes=30): day_start + pd.Timedelta(hours=9, minutes=29)]
    elif reference == "overnight":
        segment = bars.loc[day_start - pd.Timedelta(hours=6): day_start + pd.Timedelta(hours=9, minutes=29)]
    elif reference == "prior_day":
        prior_dates = sorted({d for d in bars.index.date if d < local_day})
        if not prior_dates:
            return None
        prior = prior_dates[-1]
        segment = bars[bars.index.date == prior]
    else:
        raise ValueError(reference)
    if len(segment) < 3:
        return None
    return float(segment.mid_high.max()), float(segment.mid_low.min())


def find_signal(day: pd.DataFrame, ref_high: float, ref_low: float, daily_atr: float, cfg: Config):
    active = positions(day.index, "09:30", cfg.entry_cutoff)
    if not len(active):
        return None
    high_level = ref_high + daily_atr * cfg.sweep_atr
    low_level = ref_low - daily_atr * cfg.sweep_atr
    max_wait = 0 if cfg.confirm_mode == "same_bar" else 3

    signals = []
    if cfg.side_mode != "long_only":
        for i, pos in enumerate(active):
            pos = int(pos)
            bar = day.iloc[pos]
            if bar.ask_high < high_level:
                continue
            extreme = float(bar.ask_high)
            for wait in range(max_wait + 1):
                cp = pos + wait
                if cp >= len(day) or cp > int(active[-1]):
                    break
                confirm = day.iloc[cp]
                trend_ok = (not cfg.trend_filter) or float(day.iloc[pos - 1].mid_close) > float(day.iloc[pos - 1].ema200)
                if confirm.mid_close < ref_high and trend_ok:
                    ep = cp + 1
                    if ep < len(day) and ep <= int(active[-1]):
                        signals.append((ep, -1, extreme, pos))
                    break
            break

    if cfg.side_mode != "short_only":
        for i, pos in enumerate(active):
            pos = int(pos)
            bar = day.iloc[pos]
            if bar.bid_low > low_level:
                continue
            extreme = float(bar.bid_low)
            for wait in range(max_wait + 1):
                cp = pos + wait
                if cp >= len(day) or cp > int(active[-1]):
                    break
                confirm = day.iloc[cp]
                trend_ok = (not cfg.trend_filter) or float(day.iloc[pos - 1].mid_close) < float(day.iloc[pos - 1].ema200)
                if confirm.mid_close > ref_low and trend_ok:
                    ep = cp + 1
                    if ep < len(day) and ep <= int(active[-1]):
                        signals.append((ep, 1, extreme, pos))
                    break
            break

    if not signals:
        return None
    signals.sort(key=lambda item: item[0])
    if len(signals) > 1 and signals[0][0] == signals[1][0]:
        return None
    return signals[0]


def simulate(day: pd.DataFrame, signal, daily_atr: float, cfg: Config, local_day, reference: str):
    ep, direction, extreme, sweep_pos = signal
    entry_bar = day.iloc[ep]
    stop_buffer = daily_atr * cfg.stop_buffer_atr
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

    after = day.index[ep:]
    noon = np.flatnonzero((after.hour * 60 + after.minute) >= 12 * 60)
    last = ep + int(noon[0]) if len(noon) else len(day) - 1
    exit_price = float(day.iloc[last].bid_open if direction > 0 else day.iloc[last].ask_open)
    reason = "TIME"

    for pos in range(ep, last + 1):
        bar = day.iloc[pos]
        if direction > 0:
            stop_hit = bar.bid_low <= stop
            target_hit = bar.bid_high >= target
            if stop_hit:
                exit_price = min(stop, float(bar.bid_open))
                last = pos
                reason = "STOP"
                break
            if target_hit:
                exit_price = target
                last = pos
                reason = "TP"
                break
        else:
            stop_hit = bar.ask_high >= stop
            target_hit = bar.ask_low <= target
            if stop_hit:
                exit_price = max(stop, float(bar.ask_open))
                last = pos
                reason = "STOP"
                break
            if target_hit:
                exit_price = target
                last = pos
                reason = "TP"
                break

    gross = direction * (exit_price - entry)
    net = gross - cfg.extra_cost
    entry_spread = float(entry_bar.ask_open - entry_bar.bid_open)
    ref_level = float(day.iloc[sweep_pos].mid_close)
    sweep_size = abs(extreme - ref_level) / daily_atr
    return Trade(
        date=str(local_day),
        year=int(str(local_day)[:4]),
        reference=reference,
        side="LONG" if direction > 0 else "SHORT",
        entry_time=day.index[ep].isoformat(),
        exit_time=day.index[last].isoformat(),
        entry=entry,
        exit=exit_price,
        stop=stop,
        target=target,
        risk=risk,
        net_points=net,
        r_net=net / risk,
        exit_reason=reason,
        sweep_size_atr=sweep_size,
        entry_spread=entry_spread,
    )


def backtest(bars_ny: pd.DataFrame, atr: pd.Series, cfg: Config):
    trades = []
    for local_day, day in bars_ny.groupby(bars_ny.index.date, sort=True):
        if local_day < START or local_day > END or pd.Timestamp(local_day).weekday() >= 5:
            continue
        day = day.sort_index()
        utc_day = day.index[0].tz_convert("UTC").date()
        daily_atr = prior_atr(atr, utc_day)
        if daily_atr is None or daily_atr <= 0:
            continue
        levels = reference_levels(bars_ny, local_day, cfg.reference)
        if levels is None:
            continue
        ref_high, ref_low = levels
        signal = find_signal(day, ref_high, ref_low, daily_atr, cfg)
        if signal is None:
            continue
        trade = simulate(day, signal, daily_atr, cfg, local_day, cfg.reference)
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
    bars_ny = bars_utc.tz_convert(NY)

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
    for cfg in configs:
        trades = backtest(bars_ny, atr, cfg)
        cache[cfg] = trades
        row = asdict(cfg)
        row.update({f"all_{key}": value for key, value in metrics(trades).items()})
        for year in [2024, 2025, 2026]:
            row.update({f"y{year}_{key}": value for key, value in period_metrics(trades, year).items()})
        rows.append(row)

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
        key = Config(
            str(best.reference), float(best.sweep_atr), str(best.confirm_mode), float(best.rr),
            str(best.entry_cutoff), str(best.side_mode), bool(best.trend_filter), float(best.stop_buffer_atr),
            float(best.extra_cost),
        )
        pd.DataFrame([asdict(trade) for trade in cache[key]]).to_csv(OUT / "best_discovery_trades.csv", index=False)

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
