from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data_universal")
OUT = Path("artifacts_universal")
OUT.mkdir(exist_ok=True)
ASSETS = ["xauusd", "eurusd", "btcusd"]
START = pd.Timestamp("2024-01-01", tz="UTC")
END = pd.Timestamp("2026-07-17", tz="UTC")

SESSIONS = {
    "S00": ("00:00", "00:30", "00:30", "04:00"),
    "S08": ("08:00", "08:30", "08:30", "12:00"),
    "S1330": ("13:30", "14:00", "14:00", "17:00"),
}


@dataclass
class Trade:
    asset: str
    family: str
    config: str
    side: str
    entry_time: str
    exit_time: str
    entry: float
    exit: float
    stop: float
    target: float | None
    risk: float
    r_net: float
    exit_reason: str


def read_side(asset: str, side: str) -> pd.DataFrame:
    path = DATA_DIR / f"{asset}_{side}_m1.csv"
    frame = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing {sorted(missing)}")
    index = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    prefix = side.lower()
    result = pd.DataFrame(
        {
            f"{prefix}_open": pd.to_numeric(frame.open, errors="coerce").to_numpy(),
            f"{prefix}_high": pd.to_numeric(frame.high, errors="coerce").to_numpy(),
            f"{prefix}_low": pd.to_numeric(frame.low, errors="coerce").to_numpy(),
            f"{prefix}_close": pd.to_numeric(frame.close, errors="coerce").to_numpy(),
            f"{prefix}_volume": pd.to_numeric(frame.volume, errors="coerce").to_numpy(),
        },
        index=pd.DatetimeIndex(index),
    ).dropna().sort_index()
    return result[~result.index.duplicated(keep="last")]


def load_asset(asset: str) -> pd.DataFrame:
    bid = read_side(asset, "bid")
    ask = read_side(asset, "ask")
    merged = bid.join(ask, how="inner")
    if len(merged) < 500_000:
        raise RuntimeError(f"{asset}: insufficient merged rows {len(merged)}")
    return merged


def resample_bidask(m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "bid_open": "first", "bid_high": "max", "bid_low": "min", "bid_close": "last", "bid_volume": "sum",
        "ask_open": "first", "ask_high": "max", "ask_low": "min", "ask_close": "last", "ask_volume": "sum",
    }
    bars = m1.resample(rule, label="left", closed="left").agg(agg)
    bars = bars.dropna(subset=[
        "bid_open", "bid_high", "bid_low", "bid_close",
        "ask_open", "ask_high", "ask_low", "ask_close",
    ])
    for field in ["open", "high", "low", "close"]:
        bars[f"mid_{field}"] = (bars[f"bid_{field}"] + bars[f"ask_{field}"]) / 2.0
    bars["spread_open"] = bars.ask_open - bars.bid_open
    return bars


def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def add_atr(bars: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    frame = bars.copy()
    previous = frame.mid_close.shift(1)
    tr = pd.concat(
        [
            frame.mid_high - frame.mid_low,
            (frame.mid_high - previous).abs(),
            (frame.mid_low - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr"] = rma(tr, length)
    return frame


def daily_atr_map(bars: pd.DataFrame) -> pd.Series:
    daily = bars[["mid_high", "mid_low", "mid_close"]].resample("1D").agg(
        {"mid_high": "max", "mid_low": "min", "mid_close": "last"}
    ).dropna()
    previous = daily.mid_close.shift(1)
    tr = pd.concat(
        [
            daily.mid_high - daily.mid_low,
            (daily.mid_high - previous).abs(),
            (daily.mid_low - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = rma(tr, 14).shift(1)
    atr.index = pd.Index([timestamp.date() for timestamp in atr.index])
    return atr.dropna()


def time_positions(index: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    sh, sm = map(int, start.split(":"))
    eh, em = map(int, end.split(":"))
    minutes = index.hour * 60 + index.minute
    return np.flatnonzero((minutes >= sh * 60 + sm) & (minutes < eh * 60 + em))


def first_position_at_or_after(index: pd.DatetimeIndex, clock: str) -> int | None:
    hour, minute = map(int, clock.split(":"))
    values = np.flatnonzero(index.hour * 60 + index.minute >= hour * 60 + minute)
    return int(values[0]) if len(values) else None


def metrics(trades: list[Trade]) -> dict[str, float]:
    if not trades:
        return {
            "trades": 0, "win_rate": math.nan, "profit_factor": math.nan,
            "net_r": 0.0, "avg_r": math.nan, "max_dd_r": math.nan,
            "max_consecutive_losses": 0,
        }
    rs = np.array([trade.r_net for trade in trades], dtype=float)
    gross_profit = float(rs[rs > 0].sum())
    gross_loss = float(-rs[rs < 0].sum())
    equity = np.cumsum(rs)
    peaks = np.maximum.accumulate(np.r_[0.0, equity])
    drawdowns = peaks[1:] - equity
    current = maximum = 0
    for value in rs:
        if value < 0:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return {
        "trades": int(len(trades)),
        "win_rate": float((rs > 0).mean() * 100.0),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else math.inf,
        "net_r": float(rs.sum()),
        "avg_r": float(rs.mean()),
        "max_dd_r": float(drawdowns.max()) if len(drawdowns) else 0.0,
        "max_consecutive_losses": int(maximum),
    }


def execute_fixed_exit(
    asset: str,
    family: str,
    config: str,
    day: pd.DataFrame,
    entry_pos: int,
    exit_pos: int,
    direction: int,
    stop: float,
) -> Trade | None:
    entry = float(day.iloc[entry_pos].ask_open if direction > 0 else day.iloc[entry_pos].bid_open)
    risk = entry - stop if direction > 0 else stop - entry
    if not np.isfinite(risk) or risk <= 0 or exit_pos <= entry_pos:
        return None
    exit_price = float(day.iloc[exit_pos].bid_open if direction > 0 else day.iloc[exit_pos].ask_open)
    reason = "TIME"
    actual_exit = exit_pos
    for pos in range(entry_pos, exit_pos):
        bar = day.iloc[pos]
        if direction > 0 and bar.bid_low <= stop:
            exit_price = min(float(stop), float(bar.bid_open))
            actual_exit = pos
            reason = "STOP"
            break
        if direction < 0 and bar.ask_high >= stop:
            exit_price = max(float(stop), float(bar.ask_open))
            actual_exit = pos
            reason = "STOP"
            break
    r_net = direction * (exit_price - entry) / risk
    return Trade(
        asset, family, config, "LONG" if direction > 0 else "SHORT",
        day.index[entry_pos].isoformat(), day.index[actual_exit].isoformat(),
        entry, exit_price, float(stop), None, float(risk), float(r_net), reason,
    )


def execute_rr(
    asset: str,
    family: str,
    config: str,
    bars: pd.DataFrame,
    entry_pos: int,
    last_pos: int,
    direction: int,
    stop: float,
    rr: float,
) -> tuple[Trade | None, int]:
    entry = float(bars.iloc[entry_pos].ask_open if direction > 0 else bars.iloc[entry_pos].bid_open)
    risk = entry - stop if direction > 0 else stop - entry
    if not np.isfinite(risk) or risk <= 0:
        return None, entry_pos
    target = entry + risk * rr if direction > 0 else entry - risk * rr
    exit_price = float(bars.iloc[last_pos].bid_close if direction > 0 else bars.iloc[last_pos].ask_close)
    reason = "TIME"
    actual_exit = last_pos
    for pos in range(entry_pos, last_pos + 1):
        bar = bars.iloc[pos]
        if direction > 0:
            if bar.bid_low <= stop:
                exit_price = min(float(stop), float(bar.bid_open))
                actual_exit = pos
                reason = "STOP"
                break
            if bar.bid_high >= target:
                exit_price = float(target)
                actual_exit = pos
                reason = "TP"
                break
        else:
            if bar.ask_high >= stop:
                exit_price = max(float(stop), float(bar.ask_open))
                actual_exit = pos
                reason = "STOP"
                break
            if bar.ask_low <= target:
                exit_price = float(target)
                actual_exit = pos
                reason = "TP"
                break
    r_net = direction * (exit_price - entry) / risk
    trade = Trade(
        asset, family, config, "LONG" if direction > 0 else "SHORT",
        bars.index[entry_pos].isoformat(), bars.index[actual_exit].isoformat(),
        entry, exit_price, float(stop), float(target), float(risk), float(r_net), reason,
    )
    return trade, actual_exit


def session_momentum(asset: str, bars: pd.DataFrame, daily_atr: pd.Series) -> list[Trade]:
    all_trades: list[Trade] = []
    groups = {day: frame.sort_index() for day, frame in bars.groupby(bars.index.date, sort=True)}
    for session_id, (signal_start, signal_end, entry_clock, exit_clock) in SESSIONS.items():
        stats = []
        for day_value, day in groups.items():
            positions = time_positions(day.index, signal_start, signal_end)
            if len(positions) < 4 or day_value not in daily_atr.index:
                continue
            signal = day.iloc[positions]
            atr_value = float(daily_atr.loc[day_value])
            if not np.isfinite(atr_value) or atr_value <= 0:
                continue
            stats.append(
                {
                    "day": day_value,
                    "return_atr": float((signal.iloc[-1].mid_close - signal.iloc[0].mid_open) / atr_value),
                    "range_atr": float((signal.mid_high.max() - signal.mid_low.min()) / atr_value),
                    "high": float(signal.mid_high.max()),
                    "low": float(signal.mid_low.min()),
                }
            )
        stat_frame = pd.DataFrame(stats).set_index("day").sort_index()
        if stat_frame.empty:
            continue
        stat_frame["range_median20"] = stat_frame.range_atr.rolling(20, min_periods=10).median().shift(1)
        for threshold, volume_filter in itertools.product([0.00, 0.05, 0.10], [False, True]):
            config = f"session={session_id}|retATR={threshold:.2f}|highVol={int(volume_filter)}"
            for day_value, row in stat_frame.iterrows():
                if day_value < START.date() or day_value >= END.date():
                    continue
                if abs(float(row.return_atr)) < threshold:
                    continue
                if volume_filter and (not np.isfinite(row.range_median20) or row.range_atr < row.range_median20):
                    continue
                day = groups[day_value]
                entry_pos = first_position_at_or_after(day.index, entry_clock)
                exit_pos = first_position_at_or_after(day.index, exit_clock)
                if entry_pos is None or exit_pos is None:
                    continue
                direction = 1 if row.return_atr > 0 else -1
                stop = float(row.low if direction > 0 else row.high)
                trade = execute_fixed_exit(asset, "Session momentum", config, day, entry_pos, exit_pos, direction, stop)
                if trade is not None:
                    all_trades.append(trade)
    return all_trades


def extreme_reversal(asset: str, bars: pd.DataFrame) -> list[Trade]:
    trades: list[Trade] = []
    frame = add_atr(bars, 14)
    returns = frame.mid_close.pct_change()
    frame["return_std100"] = returns.rolling(100, min_periods=100).std().shift(1)
    frame["bar_return"] = (frame.mid_close - frame.mid_open) / frame.mid_open
    frame["close_position"] = (frame.mid_close - frame.mid_low) / (frame.mid_high - frame.mid_low).replace(0, np.nan)
    for z_threshold, range_multiple, rr in itertools.product([2.5, 3.0], [1.5, 2.0], [1.0, 1.5]):
        config = f"z={z_threshold:.1f}|rangeATR={range_multiple:.1f}|RR={rr:.1f}"
        index = 101
        while index < len(frame) - 2:
            row = frame.iloc[index]
            if row.name < START or row.name >= END or not np.isfinite(row.atr) or not np.isfinite(row.return_std100):
                index += 1
                continue
            z_score = abs(float(row.bar_return)) / float(row.return_std100) if row.return_std100 > 0 else 0.0
            range_ratio = float((row.mid_high - row.mid_low) / row.atr)
            direction = 0
            if z_score >= z_threshold and range_ratio >= range_multiple:
                if row.bar_return > 0 and row.close_position <= 0.65:
                    direction = -1
                elif row.bar_return < 0 and row.close_position >= 0.35:
                    direction = 1
            if direction == 0:
                index += 1
                continue
            entry_pos = index + 1
            stop = float(row.mid_low - 0.10 * row.atr if direction > 0 else row.mid_high + 0.10 * row.atr)
            last_pos = min(entry_pos + 8, len(frame) - 1)
            trade, actual_exit = execute_rr(asset, "Extreme reversal", config, frame, entry_pos, last_pos, direction, stop, rr)
            if trade is not None:
                trades.append(trade)
                index = max(index + 1, actual_exit + 1)
            else:
                index += 1
    return trades


def compression_breakout(asset: str, bars: pd.DataFrame) -> list[Trade]:
    trades: list[Trade] = []
    daily = bars[["mid_high", "mid_low"]].resample("1D").agg({"mid_high": "max", "mid_low": "min"}).dropna()
    daily["range"] = daily.mid_high - daily.mid_low
    for n in [4, 7]:
        daily[f"nr{n}"] = daily.range.eq(daily.range.rolling(n, min_periods=n).min()).shift(1).fillna(False)
    groups = {day: frame.sort_index() for day, frame in bars.groupby(bars.index.date, sort=True)}
    daily_by_date = daily.copy()
    daily_by_date.index = pd.Index([timestamp.date() for timestamp in daily.index])
    for n, session_id, rr in itertools.product([4, 7], SESSIONS.keys(), [1.5, 2.0]):
        signal_start, signal_end, entry_clock, _ = SESSIONS[session_id]
        config = f"NR={n}|session={session_id}|RR={rr:.1f}"
        for day_value, day in groups.items():
            if day_value < START.date() or day_value >= END.date():
                continue
            if day_value not in daily_by_date.index or not bool(daily_by_date.loc[day_value, f"nr{n}"]):
                continue
            opening_positions = time_positions(day.index, signal_start, signal_end)
            if len(opening_positions) < 4:
                continue
            opening = day.iloc[opening_positions]
            or_high = float(opening.mid_high.max())
            or_low = float(opening.mid_low.min())
            if or_high <= or_low:
                continue
            entry_start = first_position_at_or_after(day.index, entry_clock)
            if entry_start is None:
                continue
            entry_deadline = min(entry_start + 24, len(day) - 1)
            found = None
            for pos in range(entry_start, entry_deadline + 1):
                bar = day.iloc[pos]
                long_hit = bar.ask_high >= or_high
                short_hit = bar.bid_low <= or_low
                if long_hit and short_hit:
                    found = None
                    break
                if long_hit:
                    found = (pos, 1, or_low, max(or_high, float(bar.ask_open)))
                    break
                if short_hit:
                    found = (pos, -1, or_high, min(or_low, float(bar.bid_open)))
                    break
            if found is None:
                continue
            entry_pos, direction, stop, custom_entry = found
            risk = custom_entry - stop if direction > 0 else stop - custom_entry
            if risk <= 0:
                continue
            target = custom_entry + risk * rr if direction > 0 else custom_entry - risk * rr
            last_pos = min(entry_pos + 48, len(day) - 1)
            exit_price = float(day.iloc[last_pos].bid_close if direction > 0 else day.iloc[last_pos].ask_close)
            actual_exit = last_pos
            reason = "TIME"
            for pos in range(entry_pos, last_pos + 1):
                bar = day.iloc[pos]
                if direction > 0:
                    if bar.bid_low <= stop:
                        exit_price = min(stop, float(bar.bid_open))
                        actual_exit = pos
                        reason = "STOP"
                        break
                    if bar.bid_high >= target:
                        exit_price = target
                        actual_exit = pos
                        reason = "TP"
                        break
                else:
                    if bar.ask_high >= stop:
                        exit_price = max(stop, float(bar.ask_open))
                        actual_exit = pos
                        reason = "STOP"
                        break
                    if bar.ask_low <= target:
                        exit_price = target
                        actual_exit = pos
                        reason = "TP"
                        break
            r_net = direction * (exit_price - custom_entry) / risk
            trades.append(
                Trade(
                    asset, "Compression breakout", config, "LONG" if direction > 0 else "SHORT",
                    day.index[entry_pos].isoformat(), day.index[actual_exit].isoformat(),
                    float(custom_entry), float(exit_price), float(stop), float(target), float(risk), float(r_net), reason,
                )
            )
    return trades


def year_of(trade: Trade) -> int:
    return pd.Timestamp(trade.entry_time).year


def build_results(all_trades: list[Trade]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades_frame = pd.DataFrame([asdict(trade) for trade in all_trades])
    rows = []
    for (family, config), subset in trades_frame.groupby(["family", "config"], sort=True):
        trade_objects = [Trade(**record) for record in subset.to_dict("records")]
        row = {"family": family, "config": config}
        row.update({f"all_{key}": value for key, value in metrics(trade_objects).items()})
        for year in [2024, 2025, 2026]:
            year_trades = [trade for trade in trade_objects if year_of(trade) == year]
            row.update({f"y{year}_{key}": value for key, value in metrics(year_trades).items()})
            positive_assets = 0
            for asset in ASSETS:
                asset_trades = [trade for trade in year_trades if trade.asset == asset]
                asset_metrics = metrics(asset_trades)
                row[f"{asset}_{year}_trades"] = asset_metrics["trades"]
                row[f"{asset}_{year}_pf"] = asset_metrics["profit_factor"]
                row[f"{asset}_{year}_avg_r"] = asset_metrics["avg_r"]
                if asset_metrics["trades"] >= 5 and asset_metrics["profit_factor"] > 1 and asset_metrics["avg_r"] > 0:
                    positive_assets += 1
            row[f"positive_assets_{year}"] = positive_assets
        rows.append(row)
    results = pd.DataFrame(rows)
    discovery = results[
        (results.y2024_trades >= 30)
        & np.isfinite(results.y2024_profit_factor)
        & (results.y2024_profit_factor > 1.0)
        & (results.y2024_avg_r > 0.0)
    ].copy()
    discovery["score"] = discovery.y2024_avg_r.clip(-1, 2) * np.sqrt(discovery.y2024_trades) - 0.03 * discovery.y2024_max_dd_r
    discovery = discovery.sort_values(["score", "y2024_profit_factor"], ascending=False)
    robust = discovery[
        (discovery.y2025_trades >= 30)
        & (discovery.y2026_trades >= 15)
        & (discovery.y2025_profit_factor > 1.0)
        & (discovery.y2026_profit_factor > 1.0)
        & (discovery.y2025_avg_r > 0.0)
        & (discovery.y2026_avg_r > 0.0)
        & (discovery.positive_assets_2025 >= 2)
        & (discovery.positive_assets_2026 >= 2)
    ].copy()
    return results, discovery, robust


def main() -> None:
    all_trades: list[Trade] = []
    data_info = []
    for asset in ASSETS:
        print(f"Loading {asset}...")
        m1 = load_asset(asset)
        bars_5m = resample_bidask(m1, "5min")
        bars_5m = bars_5m[(bars_5m.index >= pd.Timestamp("2023-11-01", tz="UTC")) & (bars_5m.index < END)]
        atr_daily = daily_atr_map(bars_5m)
        momentum = session_momentum(asset, bars_5m, atr_daily)
        reversal = extreme_reversal(asset, resample_bidask(m1, "15min"))
        compression = compression_breakout(asset, bars_5m)
        all_trades.extend(momentum)
        all_trades.extend(reversal)
        all_trades.extend(compression)
        data_info.append(
            {
                "asset": asset,
                "m1_rows": len(m1),
                "momentum_trades": len(momentum),
                "reversal_trades": len(reversal),
                "compression_trades": len(compression),
                "median_spread": float(bars_5m.spread_open.median()),
            }
        )
        print(asset, data_info[-1])

    results, discovery, robust = build_results(all_trades)
    results.to_csv(OUT / "universal_edge_results.csv", index=False)
    discovery.to_csv(OUT / "universal_edge_discovery_ranked.csv", index=False)
    robust.to_csv(OUT / "universal_edge_forward_positive.csv", index=False)
    pd.DataFrame([asdict(trade) for trade in all_trades]).to_csv(OUT / "universal_edge_all_trades.csv", index=False)
    pd.DataFrame(data_info).to_csv(OUT / "universal_edge_data_info.csv", index=False)

    display_columns = [
        "family", "config",
        "y2024_trades", "y2024_win_rate", "y2024_profit_factor", "y2024_avg_r", "y2024_max_dd_r", "positive_assets_2024",
        "y2025_trades", "y2025_win_rate", "y2025_profit_factor", "y2025_avg_r", "y2025_max_dd_r", "positive_assets_2025",
        "y2026_trades", "y2026_win_rate", "y2026_profit_factor", "y2026_avg_r", "y2026_max_dd_r", "positive_assets_2026",
        "all_trades", "all_profit_factor", "all_avg_r", "all_max_dd_r",
    ]
    report = [
        "# Universal Short-Term Edge Horse Race",
        "",
        "Assets: XAUUSD, EURUSD, BTCUSD. Data: Dukascopy BID/ASK M1, 2024-01-01 through 2026-07-16.",
        "The same parameters are applied to every asset. Results are normalized in R; longs buy ASK/sell BID and shorts sell BID/buy ASK.",
        "Families: session momentum, extreme-move reversal, and NR4/NR7 compression breakout.",
        "Selection: 2024. Validation: 2025. Forward test: 2026 YTD. A candidate must be positive on the aggregate and on at least two of three assets in both validation periods.",
        "",
        "## Data summary",
        "",
        pd.DataFrame(data_info).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Top 20 selected only on 2024",
        "",
        discovery.head(20)[display_columns].to_markdown(index=False, floatfmt=".3f") if len(discovery) else "No 2024-positive candidate.",
        "",
        "## Cross-asset forward survivors",
        "",
        f"Count: {len(robust)}",
        "",
        robust[display_columns].to_markdown(index=False, floatfmt=".3f") if len(robust) else "None passed.",
    ]
    (OUT / "universal_edge_report.md").write_text("\n".join(report), encoding="utf-8")
    print((OUT / "universal_edge_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
