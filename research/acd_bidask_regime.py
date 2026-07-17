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
OUT = Path("artifacts_regime")
OUT.mkdir(exist_ok=True)
START = pd.Timestamp("2024-01-01").date()
END = pd.Timestamp("2026-07-16").date()


@dataclass
class ContextTrade:
    date: str
    year: int
    entry_time: str
    entry_minute: int
    entry_spread: float
    pre_range_atr: float
    pre_return_atr: float
    opening_drive_atr: float
    or_close_position: float
    ema_distance_atr: float
    risk: float
    net_points: float
    r_net: float
    exit_reason: str


def prior_atr(atr: pd.Series, day) -> float | None:
    values = atr.loc[atr.index < day]
    return None if values.empty else float(values.iloc[-1])


def positions(index: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    sh, sm = map(int, start.split(":"))
    eh, em = map(int, end.split(":"))
    minute = index.hour * 60 + index.minute
    return np.flatnonzero((minute >= sh * 60 + sm) & (minute < eh * 60 + em))


def generate_trades(bars_ny: pd.DataFrame, atr: pd.Series, extra_cost: float = 0.05) -> list[ContextTrade]:
    trades: list[ContextTrade] = []
    for local_day, day in bars_ny.groupby(bars_ny.index.date, sort=True):
        if local_day < START or local_day > END:
            continue
        if pd.Timestamp(local_day).weekday() not in {1, 2, 3, 4}:
            continue
        day = day.sort_index()
        utc_day = day.index[0].tz_convert("UTC").date()
        daily_atr = prior_atr(atr, utc_day)
        if daily_atr is None or daily_atr <= 0:
            continue

        pre_pos = positions(day.index, "08:30", "09:30")
        or_pos = positions(day.index, "09:30", "09:45")
        entry_pos = positions(day.index, "09:45", "12:00")
        if len(pre_pos) < 8 or len(or_pos) < 3 or not len(entry_pos):
            continue

        pre = day.iloc[pre_pos]
        orb = day.iloc[or_pos]
        or_high = float(orb.mid_high.max())
        or_low = float(orb.mid_low.min())
        or_size = or_high - or_low
        if or_size <= 0:
            continue
        or_ratio = or_size / daily_atr
        if not 0.05 <= or_ratio <= 0.35:
            continue

        pre_range = float(pre.mid_high.max() - pre.mid_low.min()) / daily_atr
        pre_return = float(pre.iloc[-1].mid_close - pre.iloc[0].mid_open) / daily_atr
        opening_drive = float(orb.iloc[-1].mid_close - orb.iloc[0].mid_open) / daily_atr
        close_position = float((orb.iloc[-1].mid_close - or_low) / or_size)

        level = or_high + daily_atr * 0.08
        found = None
        for pos in entry_pos:
            pos = int(pos)
            if pos <= int(entry_pos[0]):
                continue
            previous = day.iloc[pos - 1]
            bar = day.iloc[pos]
            if previous.mid_close > previous.ema200 and bar.ask_high >= level:
                entry_ask = max(float(level), float(bar.ask_open))
                found = (pos, entry_ask, previous)
                break
        if found is None:
            continue

        ep, entry_ask, previous = found
        stop = or_low
        risk = entry_ask - stop
        if not np.isfinite(risk) or risk <= 0:
            continue
        target = entry_ask + risk

        after = day.index[ep:]
        noon = np.flatnonzero((after.hour * 60 + after.minute) >= 12 * 60)
        if len(noon):
            last_pos = ep + int(noon[0])
            exit_price = float(day.iloc[last_pos].bid_open)
        else:
            last_pos = len(day) - 1
            exit_price = float(day.iloc[last_pos].bid_close)
        reason = "TIME"

        for pos in range(ep, last_pos + 1):
            bar = day.iloc[pos]
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

        net = exit_price - entry_ask - extra_cost
        entry_clock = day.index[ep]
        minute_from_open = int((entry_clock.hour * 60 + entry_clock.minute) - (9 * 60 + 30))
        spread = float(day.iloc[ep].ask_open - day.iloc[ep].bid_open)
        ema_distance = float((previous.mid_close - previous.ema200) / daily_atr)
        trades.append(
            ContextTrade(
                date=str(local_day),
                year=int(str(local_day)[:4]),
                entry_time=entry_clock.isoformat(),
                entry_minute=minute_from_open,
                entry_spread=spread,
                pre_range_atr=pre_range,
                pre_return_atr=pre_return,
                opening_drive_atr=opening_drive,
                or_close_position=close_position,
                ema_distance_atr=ema_distance,
                risk=risk,
                net_points=net,
                r_net=net / risk,
                exit_reason=reason,
            )
        )
    return trades


def trade_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return metrics([])
    class T:
        pass
    objects = []
    for row in frame.itertuples():
        obj = T()
        obj.net_points = row.net_points
        obj.r_net = row.r_net
        objects.append(obj)
    return metrics(objects)


def apply_filters(
    frame: pd.DataFrame,
    entry_cutoff: int,
    pre_range_min: float,
    pre_return_min: float,
    close_position_min: float,
    spread_max: float,
) -> pd.DataFrame:
    return frame[
        (frame.entry_minute <= entry_cutoff)
        & (frame.pre_range_atr >= pre_range_min)
        & (frame.pre_return_atr >= pre_return_min)
        & (frame.or_close_position >= close_position_min)
        & (frame.entry_spread <= spread_max)
    ]


def main() -> None:
    bid = read_side(DATA_DIR / "xauusd_bid_m1.csv", "bid")
    ask = read_side(DATA_DIR / "xauusd_ask_m1.csv", "ask")
    m1 = bid.join(ask, how="inner")
    bars_utc = resample_5m(m1)
    atr = build_atr(bars_utc)
    bars_ny = bars_utc.tz_convert(NY)

    trades = generate_trades(bars_ny, atr, extra_cost=0.05)
    frame = pd.DataFrame([asdict(trade) for trade in trades])
    frame.to_csv(OUT / "base_context_trades.csv", index=False)

    baseline = []
    for year in [2024, 2025, 2026]:
        row = {"year": year}
        row.update(trade_metrics(frame[frame.year == year]))
        baseline.append(row)
    pd.DataFrame(baseline).to_csv(OUT / "baseline_by_year.csv", index=False)

    rows = []
    grid = itertools.product(
        [30, 60, 90, 150],                  # by 10:00, 10:30, 11:00, 12:00
        [0.00, 0.15, 0.25, 0.35],           # 8:30-9:30 range / ATR
        [-0.10, 0.00, 0.05, 0.10],          # 8:30-9:30 return / ATR
        [0.00, 0.50, 0.70, 0.85],           # OR close position
        [0.60, 0.80, 1.00, 99.0],           # ASK-BID spread cap
    )
    for cutoff, pre_range_min, pre_return_min, close_min, spread_max in grid:
        params = {
            "entry_cutoff": cutoff,
            "pre_range_min": pre_range_min,
            "pre_return_min": pre_return_min,
            "or_close_position_min": close_min,
            "spread_max": spread_max,
        }
        selected = apply_filters(frame, cutoff, pre_range_min, pre_return_min, close_min, spread_max)
        row = dict(params)
        for year in [2024, 2025, 2026]:
            year_metrics = trade_metrics(selected[selected.year == year])
            row.update({f"y{year}_{key}": value for key, value in year_metrics.items()})
        all_metrics = trade_metrics(selected)
        row.update({f"all_{key}": value for key, value in all_metrics.items()})
        rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "regime_grid.csv", index=False)

    discovery = results[
        (results.y2024_trades >= 30)
        & np.isfinite(results.y2024_profit_factor)
        & (results.y2024_profit_factor > 1.0)
        & (results.y2024_avg_r > 0.0)
    ].copy()
    discovery["score"] = (
        discovery.y2024_avg_r.clip(lower=-1, upper=2) * np.sqrt(discovery.y2024_trades)
        - 0.03 * discovery.y2024_max_dd_r
    )
    discovery = discovery.sort_values(["score", "y2024_profit_factor"], ascending=False)
    discovery.to_csv(OUT / "regime_discovery_ranked.csv", index=False)

    robust = discovery[
        (discovery.y2025_trades >= 25)
        & (discovery.y2026_trades >= 10)
        & (discovery.y2025_profit_factor > 1.0)
        & (discovery.y2026_profit_factor > 1.0)
        & (discovery.y2025_avg_r > 0.0)
        & (discovery.y2026_avg_r > 0.0)
    ].copy()
    robust.to_csv(OUT / "regime_forward_positive.csv", index=False)

    strict = robust[
        (robust.all_profit_factor >= 1.20)
        & (robust.all_avg_r >= 0.08)
        & (robust.all_max_dd_r <= 8.0)
    ].copy()
    strict.to_csv(OUT / "regime_strict_pass.csv", index=False)

    columns = [
        "entry_cutoff", "pre_range_min", "pre_return_min", "or_close_position_min", "spread_max",
        "y2024_trades", "y2024_win_rate", "y2024_profit_factor", "y2024_avg_r", "y2024_max_dd_r",
        "y2025_trades", "y2025_win_rate", "y2025_profit_factor", "y2025_avg_r", "y2025_max_dd_r",
        "y2026_trades", "y2026_win_rate", "y2026_profit_factor", "y2026_avg_r", "y2026_max_dd_r",
        "all_trades", "all_profit_factor", "all_avg_r", "all_max_dd_r",
    ]
    report = [
        "# XAUUSD ACD Market-Regime Filter Test",
        "",
        "Base: true BID/ASK, A=0.08 ATR, touch entry, 1R, extra cost 0.05; Tue-Fri, long only, NY OR, EMA200, no C.",
        "Filters are selected using 2024 only. 2025 and 2026 YTD remain untouched validations.",
        "",
        "## Baseline by year",
        "",
        pd.DataFrame(baseline).to_markdown(index=False, floatfmt=".3f"),
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
    (OUT / "regime_report.md").write_text("\n".join(report), encoding="utf-8")
    print((OUT / "regime_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
