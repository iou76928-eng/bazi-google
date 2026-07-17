from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data_bidask")
OUT = Path("artifacts_macd")
OUT.mkdir(exist_ok=True)
START_TEST = pd.Timestamp("2024-01-01", tz="UTC")
END_TEST = pd.Timestamp("2026-07-17", tz="UTC")


@dataclass(frozen=True)
class Variant:
    name: str
    require_sweep: bool
    require_mss: bool
    min_chain: int = 1
    risk_reward: float = 1.5
    extra_cost: float = 0.0


@dataclass
class Trade:
    timeframe: str
    variant: str
    side: str
    signal_time: str
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
    chain: int


def read_side(path: Path, side: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
    index = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    prefix = side.lower()
    result = pd.DataFrame(
        {
            f"{prefix}_open": pd.to_numeric(frame["open"], errors="coerce").to_numpy(),
            f"{prefix}_high": pd.to_numeric(frame["high"], errors="coerce").to_numpy(),
            f"{prefix}_low": pd.to_numeric(frame["low"], errors="coerce").to_numpy(),
            f"{prefix}_close": pd.to_numeric(frame["close"], errors="coerce").to_numpy(),
            f"{prefix}_volume": pd.to_numeric(frame["volume"], errors="coerce").to_numpy(),
        },
        index=pd.DatetimeIndex(index),
    )
    result = result.dropna().sort_index()
    return result[~result.index.duplicated(keep="last")]


def resample_bidask(m1: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    agg = {
        "bid_open": "first", "bid_high": "max", "bid_low": "min", "bid_close": "last", "bid_volume": "sum",
        "ask_open": "first", "ask_high": "max", "ask_low": "min", "ask_close": "last", "ask_volume": "sum",
    }
    bars = m1.resample(timeframe, label="left", closed="left").agg(agg)
    bars = bars.dropna(subset=[
        "bid_open", "bid_high", "bid_low", "bid_close",
        "ask_open", "ask_high", "ask_low", "ask_close",
    ])
    for field in ["open", "high", "low", "close"]:
        bars[f"mid_{field}"] = (bars[f"bid_{field}"] + bars[f"ask_{field}"]) / 2.0
    bars["spread_open"] = bars["ask_open"] - bars["bid_open"]
    return bars


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=1).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=1).mean()


def add_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.copy()
    fast = ema(frame.mid_close, 13)
    slow = ema(frame.mid_close, 34)
    macd = fast - slow
    frame["hist"] = macd - ema(macd, 9)
    previous = frame.mid_close.shift(1)
    true_range = pd.concat(
        [
            frame.mid_high - frame.mid_low,
            (frame.mid_high - previous).abs(),
            (frame.mid_low - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr"] = rma(true_range, 13)
    return frame


def pivot_flags(values: np.ndarray, left: int = 3, right: int = 3, kind: str = "low") -> np.ndarray:
    flags = np.zeros(len(values), dtype=bool)
    for index in range(left, len(values) - right):
        window = values[index - left : index + right + 1]
        if not np.all(np.isfinite(window)):
            continue
        flags[index] = values[index] <= np.min(window) if kind == "low" else values[index] >= np.max(window)
    return flags


def trade_metrics(trades: list[Trade]) -> dict[str, float]:
    if not trades:
        return {
            "trades": 0, "win_rate": math.nan, "profit_factor": math.nan,
            "net_points": 0.0, "avg_r": math.nan, "max_dd_r": math.nan,
            "max_consecutive_losses": 0,
        }
    profit = np.array([trade.net_points for trade in trades], dtype=float)
    rs = np.array([trade.r_net for trade in trades], dtype=float)
    gross_profit = float(profit[profit > 0].sum())
    gross_loss = float(-profit[profit < 0].sum())
    equity = np.cumsum(rs)
    peaks = np.maximum.accumulate(np.r_[0.0, equity])
    drawdown = peaks[1:] - equity
    streak = maximum_streak = 0
    for value in profit:
        if value < 0:
            streak += 1
            maximum_streak = max(maximum_streak, streak)
        else:
            streak = 0
    return {
        "trades": int(len(trades)),
        "win_rate": float((profit > 0).mean() * 100.0),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else math.inf,
        "net_points": float(profit.sum()),
        "avg_r": float(rs.mean()),
        "max_dd_r": float(drawdown.max()) if len(drawdown) else 0.0,
        "max_consecutive_losses": int(maximum_streak),
    }


def year_metrics(trades: list[Trade], year: int) -> dict[str, float]:
    return trade_metrics([trade for trade in trades if pd.Timestamp(trade.entry_time).year == year])


def simulate_variant(bars: pd.DataFrame, timeframe_name: str, variant: Variant) -> list[Trade]:
    count = len(bars)
    if count < 100:
        return []

    low = bars.mid_low.to_numpy(float)
    high = bars.mid_high.to_numpy(float)
    close = bars.mid_close.to_numpy(float)
    histogram = bars.hist.to_numpy(float)
    atr = bars.atr.to_numpy(float)
    bid_open = bars.bid_open.to_numpy(float)
    bid_high = bars.bid_high.to_numpy(float)
    bid_low = bars.bid_low.to_numpy(float)
    bid_close = bars.bid_close.to_numpy(float)
    ask_open = bars.ask_open.to_numpy(float)
    ask_high = bars.ask_high.to_numpy(float)
    ask_low = bars.ask_low.to_numpy(float)
    ask_close = bars.ask_close.to_numpy(float)
    timestamps = bars.index

    pivot_low = pivot_flags(low, kind="low")
    pivot_high = pivot_flags(high, kind="high")

    previous_low_price = previous_low_histogram = math.nan
    previous_low_bar = None
    previous_low_chain = 0
    previous_high_price = previous_high_histogram = math.nan
    previous_high_bar = None
    previous_high_chain = 0

    pending_long = pending_short = False
    pending_long_stop = pending_long_mss = math.nan
    pending_short_stop = pending_short_mss = math.nan
    pending_long_expiry = pending_short_expiry = -1
    pending_long_chain = pending_short_chain = 0
    pending_long_signal_time = pending_short_signal_time = None

    submitted_side = 0
    submitted_stop = math.nan
    submitted_chain = 0
    submitted_signal_time = None
    submitted_bar = -1

    position_side = 0
    entry_price = stop_price = target_price = initial_risk = math.nan
    entry_time = signal_time = None
    active_chain = 0
    trades: list[Trade] = []

    for bar in range(count):
        # Market orders submitted on the prior close fill at this bar's open.
        if position_side == 0 and submitted_side != 0 and bar == submitted_bar + 1:
            fill = float(ask_open[bar] if submitted_side > 0 else bid_open[bar])
            risk = fill - submitted_stop if submitted_side > 0 else submitted_stop - fill
            if np.isfinite(risk) and risk > 0:
                position_side = submitted_side
                entry_price = fill
                stop_price = float(submitted_stop)
                initial_risk = float(risk)
                target_price = fill + risk * variant.risk_reward if submitted_side > 0 else fill - risk * variant.risk_reward
                entry_time = timestamps[bar]
                signal_time = submitted_signal_time
                active_chain = submitted_chain
            submitted_side = 0
            submitted_bar = -1

        exited_this_bar = False
        if position_side > 0:
            if bid_low[bar] <= stop_price:
                exit_price = min(float(stop_price), float(bid_open[bar]))
                reason = "STOP"
                exited_this_bar = True
            elif bid_high[bar] >= target_price:
                exit_price = float(target_price)
                reason = "TP"
                exited_this_bar = True
            if exited_this_bar:
                net = exit_price - entry_price - variant.extra_cost
                trades.append(Trade(
                    timeframe_name, variant.name, "LONG", pd.Timestamp(signal_time).isoformat(),
                    pd.Timestamp(entry_time).isoformat(), timestamps[bar].isoformat(),
                    float(entry_price), float(exit_price), float(stop_price), float(target_price),
                    float(initial_risk), float(net), float(net / initial_risk), reason, int(active_chain),
                ))
                position_side = 0
        elif position_side < 0:
            if ask_high[bar] >= stop_price:
                exit_price = max(float(stop_price), float(ask_open[bar]))
                reason = "STOP"
                exited_this_bar = True
            elif ask_low[bar] <= target_price:
                exit_price = float(target_price)
                reason = "TP"
                exited_this_bar = True
            if exited_this_bar:
                net = entry_price - exit_price - variant.extra_cost
                trades.append(Trade(
                    timeframe_name, variant.name, "SHORT", pd.Timestamp(signal_time).isoformat(),
                    pd.Timestamp(entry_time).isoformat(), timestamps[bar].isoformat(),
                    float(entry_price), float(exit_price), float(stop_price), float(target_price),
                    float(initial_risk), float(net), float(net / initial_risk), reason, int(active_chain),
                ))
                position_side = 0

        pivot_index = bar - 3
        long_signal = short_signal = False
        long_stop = long_mss = short_stop = short_mss = math.nan
        long_chain = short_chain = 0

        # Pine processes pivot lows first, then pivot highs.
        if pivot_index >= 3 and pivot_low[pivot_index] and np.isfinite(histogram[pivot_index]) and np.isfinite(atr[pivot_index]):
            has_previous = previous_low_bar is not None and np.isfinite(previous_low_price) and np.isfinite(previous_low_histogram)
            gap_ok = has_previous and pivot_index - int(previous_low_bar) <= 120
            price_lower = has_previous and low[pivot_index] < previous_low_price
            hist_higher = has_previous and histogram[pivot_index] > previous_low_histogram
            hist_side_ok = has_previous and histogram[pivot_index] < 0 and previous_low_histogram < 0
            sweep_ok = (not variant.require_sweep) or (has_previous and close[pivot_index] > previous_low_price)
            divergence = bool(gap_ok and price_lower and hist_higher and hist_side_ok and sweep_ok)
            current_chain = previous_low_chain + 1 if divergence else 0
            if divergence and current_chain >= variant.min_chain and pivot_index >= 5:
                long_signal = True
                long_chain = current_chain
                long_stop = low[pivot_index] - atr[pivot_index]
                long_mss = float(np.max(high[pivot_index - 5 : pivot_index]))
            previous_low_price = float(low[pivot_index])
            previous_low_histogram = float(histogram[pivot_index])
            previous_low_bar = pivot_index
            previous_low_chain = current_chain

        if pivot_index >= 3 and pivot_high[pivot_index] and np.isfinite(histogram[pivot_index]) and np.isfinite(atr[pivot_index]):
            has_previous = previous_high_bar is not None and np.isfinite(previous_high_price) and np.isfinite(previous_high_histogram)
            gap_ok = has_previous and pivot_index - int(previous_high_bar) <= 120
            price_higher = has_previous and high[pivot_index] > previous_high_price
            hist_lower = has_previous and histogram[pivot_index] < previous_high_histogram
            hist_side_ok = has_previous and histogram[pivot_index] > 0 and previous_high_histogram > 0
            sweep_ok = (not variant.require_sweep) or (has_previous and close[pivot_index] < previous_high_price)
            divergence = bool(gap_ok and price_higher and hist_lower and hist_side_ok and sweep_ok)
            current_chain = previous_high_chain + 1 if divergence else 0
            if divergence and current_chain >= variant.min_chain and pivot_index >= 5:
                short_signal = True
                short_chain = current_chain
                short_stop = high[pivot_index] + atr[pivot_index]
                short_mss = float(np.min(low[pivot_index - 5 : pivot_index]))
            previous_high_price = float(high[pivot_index])
            previous_high_histogram = float(histogram[pivot_index])
            previous_high_bar = pivot_index
            previous_high_chain = current_chain

        if long_signal:
            pending_long = True
            pending_long_stop = long_stop
            pending_long_mss = long_mss
            pending_long_expiry = bar + 20
            pending_long_chain = long_chain
            pending_long_signal_time = timestamps[bar]
            pending_short = False
        if short_signal:
            pending_short = True
            pending_short_stop = short_stop
            pending_short_mss = short_mss
            pending_short_expiry = bar + 20
            pending_short_chain = short_chain
            pending_short_signal_time = timestamps[bar]
            pending_long = False

        if pending_long and bar > pending_long_expiry:
            pending_long = False
        if pending_short and bar > pending_short_expiry:
            pending_short = False

        can_submit = position_side == 0 and submitted_side == 0 and not exited_this_bar
        if can_submit and pending_long:
            distance_ok = np.isfinite(pending_long_stop) and close[bar] - pending_long_stop <= atr[bar] * 2.0
            mss_ok = (not variant.require_mss) or close[bar] > pending_long_mss
            if distance_ok and mss_ok:
                submitted_side = 1
                submitted_stop = float(pending_long_stop)
                submitted_chain = int(pending_long_chain)
                submitted_signal_time = pending_long_signal_time
                submitted_bar = bar
                pending_long = False

        can_submit = position_side == 0 and submitted_side == 0 and not exited_this_bar
        if can_submit and pending_short:
            distance_ok = np.isfinite(pending_short_stop) and pending_short_stop - close[bar] <= atr[bar] * 2.0
            mss_ok = (not variant.require_mss) or close[bar] < pending_short_mss
            if distance_ok and mss_ok:
                submitted_side = -1
                submitted_stop = float(pending_short_stop)
                submitted_chain = int(pending_short_chain)
                submitted_signal_time = pending_short_signal_time
                submitted_bar = bar
                pending_short = False

    if position_side > 0:
        exit_price = float(bid_close[-1])
        net = exit_price - entry_price - variant.extra_cost
        trades.append(Trade(timeframe_name, variant.name, "LONG", pd.Timestamp(signal_time).isoformat(), pd.Timestamp(entry_time).isoformat(), timestamps[-1].isoformat(), float(entry_price), exit_price, float(stop_price), float(target_price), float(initial_risk), float(net), float(net / initial_risk), "END", int(active_chain)))
    elif position_side < 0:
        exit_price = float(ask_close[-1])
        net = entry_price - exit_price - variant.extra_cost
        trades.append(Trade(timeframe_name, variant.name, "SHORT", pd.Timestamp(signal_time).isoformat(), pd.Timestamp(entry_time).isoformat(), timestamps[-1].isoformat(), float(entry_price), exit_price, float(stop_price), float(target_price), float(initial_risk), float(net), float(net / initial_risk), "END", int(active_chain)))

    return trades


def main() -> None:
    bid = read_side(DATA_DIR / "xauusd_bid_m1.csv", "bid")
    ask = read_side(DATA_DIR / "xauusd_ask_m1.csv", "ask")
    minute_data = bid.join(ask, how="inner")
    if len(minute_data) < 500_000:
        raise RuntimeError(f"Insufficient merged BID/ASK rows: {len(minute_data)}")

    variants = [
        Variant("Pure divergence", False, False),
        Variant("Divergence + sweep", True, False),
        Variant("Divergence + MSS", False, True),
        Variant("Default sweep + MSS", True, True),
        Variant("Default chain 2", True, True, min_chain=2),
        Variant("Default cost +0.10", True, True, extra_cost=0.10),
    ]
    timeframes = [("5min", "5m"), ("15min", "15m")]

    rows = []
    all_trades = []
    for rule, label in timeframes:
        bars = add_indicators(resample_bidask(minute_data, rule))
        bars = bars[(bars.index >= START_TEST) & (bars.index < END_TEST)]
        for variant in variants:
            trades = simulate_variant(bars, label, variant)
            all_trades.extend(trades)
            row = {"timeframe": label, **asdict(variant)}
            row.update({f"all_{key}": value for key, value in trade_metrics(trades).items()})
            for year in [2024, 2025, 2026]:
                row.update({f"y{year}_{key}": value for key, value in year_metrics(trades, year).items()})
            rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "macd_variant_results.csv", index=False)
    pd.DataFrame([asdict(trade) for trade in all_trades]).to_csv(OUT / "macd_all_trades.csv", index=False)
    default = results[results.name == "Default sweep + MSS"].copy()
    default.to_csv(OUT / "macd_default_results.csv", index=False)

    positive = results[
        (results.y2024_trades >= 20)
        & (results.y2025_trades >= 20)
        & (results.y2026_trades >= 8)
        & (results.y2024_profit_factor > 1.0)
        & (results.y2025_profit_factor > 1.0)
        & (results.y2026_profit_factor > 1.0)
        & (results.y2024_avg_r > 0.0)
        & (results.y2025_avg_r > 0.0)
        & (results.y2026_avg_r > 0.0)
    ].copy()
    positive.to_csv(OUT / "macd_cross_year_positive.csv", index=False)

    columns = [
        "timeframe", "name", "require_sweep", "require_mss", "min_chain", "risk_reward", "extra_cost",
        "y2024_trades", "y2024_win_rate", "y2024_profit_factor", "y2024_avg_r", "y2024_max_dd_r",
        "y2025_trades", "y2025_win_rate", "y2025_profit_factor", "y2025_avg_r", "y2025_max_dd_r",
        "y2026_trades", "y2026_win_rate", "y2026_profit_factor", "y2026_avg_r", "y2026_max_dd_r",
        "all_trades", "all_win_rate", "all_profit_factor", "all_avg_r", "all_max_dd_r",
    ]
    report = [
        "# MACD 13/34/9 Divergence × SMC × ATR Backtest",
        "",
        "- Data: Dukascopy XAUUSD BID/ASK M1, resampled to 5m and 15m.",
        "- Period: 2024-01-01 through 2026-07-16.",
        "- Signals use mid-price OHLC; longs enter on ASK and exit on BID; shorts enter on BID and exit on ASK.",
        "- Pivot left/right = 3/3. Signals are emitted only after all right-side confirmation bars complete; entries fill at the next bar open.",
        "- Default: histogram on the correct zero-axis side, sweep-and-reclaim, MSS, ATR13 stop, 1.5R target, 20-bar MSS expiry and 2 ATR chase limit.",
        "- Same-bar stop/target ambiguity is counted as stop. Spread is included; the stress variant adds 0.10 gold points per round trip.",
        "",
        "## Default Pine settings",
        "",
        default[columns].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Ablation comparison",
        "",
        results[columns].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Positive in 2024, 2025 and 2026 YTD",
        "",
        f"Count: {len(positive)}",
        "",
        positive[columns].to_markdown(index=False, floatfmt=".3f") if len(positive) else "None passed.",
    ]
    (OUT / "macd_backtest_report.md").write_text("\n".join(report), encoding="utf-8")
    print((OUT / "macd_backtest_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
