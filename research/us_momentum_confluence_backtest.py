from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data_us_momentum")
OUT = Path("artifacts_us_momentum")
OUT.mkdir(exist_ok=True)

TARGETS = ["xauusd", "eurusd", "gbpusd", "usdjpy"]
CONFIRMERS = ["dollaridxusd", "spyususd"]
START = pd.Timestamp("2018-01-01", tz="UTC")
END = pd.Timestamp("2026-07-17", tz="UTC")
NY = "America/New_York"

PERIODS = {
    "discovery": (2018, 2022),
    "validation": (2023, 2024),
    "forward": (2025, 2026),
    "all": (2018, 2026),
}


@dataclass(frozen=True)
class Config:
    session: str
    range_atr: float
    rr: float
    trigger_bars: int
    dxy_confirm: bool
    dxy_min_atr: float
    spy_confirm: bool

    @property
    def name(self) -> str:
        return (
            f"session={self.session}|rangeATR={self.range_atr:.2f}|RR={self.rr:.1f}"
            f"|trigger={self.trigger_bars}|DXY={int(self.dxy_confirm)}"
            f"|DXYmin={self.dxy_min_atr:.2f}|SPY={int(self.spy_confirm)}"
        )


@dataclass
class Trade:
    asset: str
    config: str
    session: str
    side: str
    signal_time: str
    entry_time: str
    exit_time: str
    entry: float
    exit: float
    stop: float
    target: float
    risk: float
    r_net: float
    exit_reason: str
    signal_range_atr: float
    dxy_move_atr: float
    spy_direction: int


def read_side(asset: str, side: str) -> pd.DataFrame:
    path = DATA / f"{asset}_{side}_m5.csv"
    frame = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns {sorted(missing)}")
    index = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    result = pd.DataFrame(
        {
            f"{side}_open": pd.to_numeric(frame["open"], errors="coerce").to_numpy(),
            f"{side}_high": pd.to_numeric(frame["high"], errors="coerce").to_numpy(),
            f"{side}_low": pd.to_numeric(frame["low"], errors="coerce").to_numpy(),
            f"{side}_close": pd.to_numeric(frame["close"], errors="coerce").to_numpy(),
            f"{side}_volume": pd.to_numeric(frame["volume"], errors="coerce").to_numpy(),
        },
        index=pd.DatetimeIndex(index),
    ).dropna().sort_index()
    return result[~result.index.duplicated(keep="last")]


def load_asset(asset: str) -> pd.DataFrame:
    bars = read_side(asset, "bid").join(read_side(asset, "ask"), how="inner")
    bars = bars[(bars.index >= START) & (bars.index < END)]
    if len(bars) < 100_000:
        raise RuntimeError(f"{asset}: insufficient merged rows {len(bars)}")
    for field in ["open", "high", "low", "close"]:
        bars[f"mid_{field}"] = (bars[f"bid_{field}"] + bars[f"ask_{field}"]) / 2.0
    previous = bars.mid_close.shift(1)
    true_range = pd.concat(
        [
            bars.mid_high - bars.mid_low,
            (bars.mid_high - previous).abs(),
            (bars.mid_low - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    bars["atr20"] = true_range.rolling(20, min_periods=20).mean().shift(1)
    bars["ny_time"] = bars.index.tz_convert(NY)
    bars["ny_date"] = bars.ny_time.dt.date
    bars["ny_minute"] = bars.ny_time.dt.hour * 60 + bars.ny_time.dt.minute
    return bars


def sign(value: float, tolerance: float = 0.0) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def dollar_direction(asset: str, target_direction: int) -> int:
    if asset in {"xauusd", "eurusd", "gbpusd"}:
        return -target_direction
    if asset == "usdjpy":
        return target_direction
    raise ValueError(asset)


def spy_expected_direction(asset: str, target_direction: int) -> int | None:
    # Risk-on mapping used only as a testable hypothesis.
    if asset in {"eurusd", "gbpusd", "usdjpy"}:
        return target_direction
    return None


def exact_bar(day: pd.DataFrame, minute: int):
    found = day[day.ny_minute == minute]
    return None if found.empty else found.iloc[0]


def pre_window(day: pd.DataFrame, start_minute: int, end_minute: int):
    frame = day[(day.ny_minute >= start_minute) & (day.ny_minute < end_minute)]
    return frame if len(frame) >= int((end_minute - start_minute) / 5 * 0.75) else None


def execute_trade(
    asset: str,
    config: Config,
    day: pd.DataFrame,
    signal_row: pd.Series,
    signal_direction: int,
    signal_range_atr: float,
    dxy_move_atr: float,
    spy_direction: int,
    force_exit_minute: int,
):
    signal_minute = int(signal_row.ny_minute)
    after_signal = day[day.ny_minute > signal_minute]
    trigger_rows = after_signal.head(config.trigger_bars)
    if trigger_rows.empty:
        return None

    level = float(signal_row.mid_high if signal_direction > 0 else signal_row.mid_low)
    stop = float(signal_row.mid_low if signal_direction > 0 else signal_row.mid_high)

    entry_time = None
    entry = None
    entry_index = None
    for timestamp, row in trigger_rows.iterrows():
        if signal_direction > 0 and row.ask_high >= level:
            entry = max(level, float(row.ask_open))
            entry_time = timestamp
            entry_index = day.index.get_loc(timestamp)
            break
        if signal_direction < 0 and row.bid_low <= level:
            entry = min(level, float(row.bid_open))
            entry_time = timestamp
            entry_index = day.index.get_loc(timestamp)
            break
    if entry is None:
        return None

    risk = entry - stop if signal_direction > 0 else stop - entry
    if not np.isfinite(risk) or risk <= 0:
        return None
    target = entry + risk * config.rr if signal_direction > 0 else entry - risk * config.rr

    exit_price = None
    exit_time = None
    reason = None
    for position in range(entry_index, len(day)):
        row = day.iloc[position]
        minute = int(row.ny_minute)
        timestamp = day.index[position]
        if minute >= force_exit_minute:
            exit_price = float(row.bid_open if signal_direction > 0 else row.ask_open)
            exit_time = timestamp
            reason = "TIME"
            break
        if signal_direction > 0:
            if row.bid_low <= stop:
                exit_price = min(stop, float(row.bid_open))
                exit_time = timestamp
                reason = "STOP"
                break
            if row.bid_high >= target:
                exit_price = target
                exit_time = timestamp
                reason = "TP"
                break
        else:
            if row.ask_high >= stop:
                exit_price = max(stop, float(row.ask_open))
                exit_time = timestamp
                reason = "STOP"
                break
            if row.ask_low <= target:
                exit_price = target
                exit_time = timestamp
                reason = "TP"
                break

    if exit_price is None:
        last = day.iloc[-1]
        exit_price = float(last.bid_close if signal_direction > 0 else last.ask_close)
        exit_time = day.index[-1]
        reason = "END"

    r_net = signal_direction * (exit_price - entry) / risk
    return Trade(
        asset=asset,
        config=config.name,
        session=config.session,
        side="LONG" if signal_direction > 0 else "SHORT",
        signal_time=signal_row.name.isoformat(),
        entry_time=entry_time.isoformat(),
        exit_time=exit_time.isoformat(),
        entry=float(entry),
        exit=float(exit_price),
        stop=float(stop),
        target=float(target),
        risk=float(risk),
        r_net=float(r_net),
        exit_reason=reason,
        signal_range_atr=float(signal_range_atr),
        dxy_move_atr=float(dxy_move_atr),
        spy_direction=int(spy_direction),
    )


def run_config_for_asset(
    asset: str,
    target: pd.DataFrame,
    dxy: pd.DataFrame,
    spy: pd.DataFrame,
    config: Config,
):
    if config.session == "0830":
        pre_start, signal_minute, force_exit = 7 * 60 + 30, 8 * 60 + 30, 11 * 60
    else:
        pre_start, signal_minute, force_exit = 8 * 60 + 30, 9 * 60 + 30, 12 * 60

    target_days = {date: frame for date, frame in target.groupby("ny_date", sort=True)}
    dxy_days = {date: frame for date, frame in dxy.groupby("ny_date", sort=True)}
    spy_days = {date: frame for date, frame in spy.groupby("ny_date", sort=True)}
    trades = []

    for date, day in target_days.items():
        if date not in dxy_days:
            continue
        pre = pre_window(day, pre_start, signal_minute)
        signal_row = exact_bar(day, signal_minute)
        dxy_row = exact_bar(dxy_days[date], signal_minute)
        spy_row = exact_bar(spy_days.get(date, pd.DataFrame()), signal_minute) if date in spy_days else None
        if pre is None or signal_row is None or dxy_row is None:
            continue
        if not np.isfinite(signal_row.atr20) or signal_row.atr20 <= 0:
            continue

        pre_high = float(pre.mid_high.max())
        pre_low = float(pre.mid_low.min())
        long_break = signal_row.mid_high > pre_high
        short_break = signal_row.mid_low < pre_low
        if long_break == short_break:
            continue

        direction = 1 if long_break else -1
        bar_range = float(signal_row.mid_high - signal_row.mid_low)
        range_atr = bar_range / float(signal_row.atr20)
        if range_atr < config.range_atr or bar_range <= 0:
            continue
        close_position = float((signal_row.mid_close - signal_row.mid_low) / bar_range)
        if direction > 0 and close_position < 0.75:
            continue
        if direction < 0 and close_position > 0.25:
            continue

        dxy_range = float(dxy_row.mid_high - dxy_row.mid_low)
        dxy_move = float(dxy_row.mid_close - dxy_row.mid_open)
        dxy_atr = float(dxy_row.atr20) if np.isfinite(dxy_row.atr20) else math.nan
        dxy_move_atr = abs(dxy_move) / dxy_atr if np.isfinite(dxy_atr) and dxy_atr > 0 else 0.0
        if config.dxy_confirm:
            expected = dollar_direction(asset, direction)
            if sign(dxy_move) != expected or dxy_move_atr < config.dxy_min_atr:
                continue

        spy_direction = 0
        if spy_row is not None:
            spy_direction = sign(float(spy_row.mid_close - spy_row.mid_open))
        if config.spy_confirm:
            expected_spy = spy_expected_direction(asset, direction)
            if expected_spy is None or spy_direction != expected_spy:
                continue

        trade = execute_trade(
            asset,
            config,
            day,
            signal_row,
            direction,
            range_atr,
            dxy_move_atr,
            spy_direction,
            force_exit,
        )
        if trade is not None:
            trades.append(trade)
    return trades


def metrics(trades: list[Trade]):
    if not trades:
        return {
            "trades": 0,
            "win_rate": math.nan,
            "profit_factor": math.nan,
            "net_r": 0.0,
            "avg_r": math.nan,
            "max_dd_r": math.nan,
            "max_consecutive_losses": 0,
        }
    ordered = sorted(trades, key=lambda trade: trade.exit_time)
    values = np.array([trade.r_net for trade in ordered], dtype=float)
    gross_profit = float(values[values > 0].sum())
    gross_loss = float(-values[values < 0].sum())
    equity = np.cumsum(values)
    peaks = np.maximum.accumulate(np.r_[0.0, equity])
    drawdowns = peaks[1:] - equity
    streak = maximum = 0
    for value in values:
        if value < 0:
            streak += 1
            maximum = max(maximum, streak)
        else:
            streak = 0
    return {
        "trades": int(len(values)),
        "win_rate": float((values > 0).mean() * 100.0),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else math.inf,
        "net_r": float(values.sum()),
        "avg_r": float(values.mean()),
        "max_dd_r": float(drawdowns.max()) if len(drawdowns) else 0.0,
        "max_consecutive_losses": int(maximum),
    }


def period_subset(trades: list[Trade], start_year: int, end_year: int):
    return [
        trade
        for trade in trades
        if start_year <= pd.Timestamp(trade.entry_time).year <= end_year
    ]


def positive_asset_count(trades: list[Trade], start_year: int, end_year: int):
    count = 0
    for asset in TARGETS:
        result = metrics([trade for trade in period_subset(trades, start_year, end_year) if trade.asset == asset])
        if result["trades"] >= 5 and result["profit_factor"] > 1.0 and result["avg_r"] > 0:
            count += 1
    return count


def build_configs():
    configs = []
    for range_atr, rr, trigger, dxy_confirm, dxy_min in itertools.product(
        [1.0, 1.5, 2.0], [1.5, 2.0], [1, 3], [False, True], [0.0, 0.25]
    ):
        if not dxy_confirm and dxy_min > 0:
            continue
        configs.append(Config("0830", range_atr, rr, trigger, dxy_confirm, dxy_min, False))
    for range_atr, rr, trigger, dxy_confirm, spy_confirm in itertools.product(
        [1.0, 1.5], [1.5, 2.0], [1, 3], [False, True], [False, True]
    ):
        configs.append(Config("0930", range_atr, rr, trigger, dxy_confirm, 0.0, spy_confirm))
    return configs


def main():
    data = {asset: load_asset(asset) for asset in TARGETS + CONFIRMERS}
    data_info = []
    for asset, bars in data.items():
        data_info.append(
            {
                "asset": asset,
                "rows": len(bars),
                "start": bars.index.min().isoformat(),
                "end": bars.index.max().isoformat(),
                "median_spread": float((bars.ask_open - bars.bid_open).median()),
            }
        )

    configs = build_configs()
    all_trades = []
    rows = []
    for config in configs:
        config_trades = []
        for asset in TARGETS:
            trades = run_config_for_asset(asset, data[asset], data["dollaridxusd"], data["spyususd"], config)
            config_trades.extend(trades)
            all_trades.extend(trades)
        row = asdict(config)
        row["config"] = config.name
        for period_name, (start_year, end_year) in PERIODS.items():
            result = metrics(period_subset(config_trades, start_year, end_year))
            row.update({f"{period_name}_{key}": value for key, value in result.items()})
            row[f"{period_name}_positive_assets"] = positive_asset_count(config_trades, start_year, end_year)
            for asset in TARGETS:
                asset_result = metrics(
                    [
                        trade
                        for trade in period_subset(config_trades, start_year, end_year)
                        if trade.asset == asset
                    ]
                )
                row[f"{period_name}_{asset}_trades"] = asset_result["trades"]
                row[f"{period_name}_{asset}_pf"] = asset_result["profit_factor"]
                row[f"{period_name}_{asset}_avg_r"] = asset_result["avg_r"]
        rows.append(row)

    results = pd.DataFrame(rows)
    results["discovery_score"] = (
        results.discovery_avg_r.fillna(-10.0) * np.sqrt(results.discovery_trades.clip(lower=1))
        - 0.03 * results.discovery_max_dd_r.fillna(100.0)
    )
    discovery = results[
        (results.discovery_trades >= 60)
        & (results.discovery_profit_factor > 1.0)
        & (results.discovery_avg_r > 0.0)
    ].sort_values(["discovery_score", "discovery_profit_factor"], ascending=False)
    robust = discovery[
        (discovery.validation_trades >= 25)
        & (discovery.forward_trades >= 15)
        & (discovery.validation_profit_factor > 1.0)
        & (discovery.forward_profit_factor > 1.0)
        & (discovery.validation_avg_r > 0.0)
        & (discovery.forward_avg_r > 0.0)
        & (discovery.validation_positive_assets >= 2)
        & (discovery.forward_positive_assets >= 2)
    ].copy()

    results.to_csv(OUT / "us_momentum_grid.csv", index=False)
    discovery.to_csv(OUT / "us_momentum_discovery_ranked.csv", index=False)
    robust.to_csv(OUT / "us_momentum_robust_candidates.csv", index=False)
    pd.DataFrame([asdict(trade) for trade in all_trades]).to_csv(OUT / "us_momentum_all_trades.csv", index=False)
    pd.DataFrame(data_info).to_csv(OUT / "us_momentum_data_info.csv", index=False)

    display_columns = [
        "config", "session",
        "discovery_trades", "discovery_profit_factor", "discovery_avg_r", "discovery_max_dd_r", "discovery_positive_assets",
        "validation_trades", "validation_profit_factor", "validation_avg_r", "validation_max_dd_r", "validation_positive_assets",
        "forward_trades", "forward_profit_factor", "forward_avg_r", "forward_max_dd_r", "forward_positive_assets",
        "all_trades", "all_profit_factor", "all_avg_r", "all_max_dd_r",
    ]
    report = [
        "# US Session 5-Minute Momentum Confluence Backtest",
        "",
        "Targets: XAUUSD, EURUSD, GBPUSD, USDJPY. Confirmers: US Dollar Index and SPY.",
        "Data: Dukascopy BID/ASK M5, 2018-01-01 through 2026-07-16.",
        "Discovery: 2018-2022. Validation: 2023-2024. Forward: 2025-2026 YTD.",
        "8:30 model: 7:30-8:25 New York range, 8:30 shock bar, optional DXY direction confirmation, trigger during next 1 or 3 bars, exit by 11:00.",
        "9:30 model: 8:30-9:25 range, 9:30 shock bar, optional DXY and SPY direction confirmation, trigger during next 1 or 3 bars, exit by noon.",
        "Signals require one-sided range breakout, signal range threshold versus prior ATR20, and close in the outer 25% of the bar.",
        "Longs enter on ASK and exit on BID; shorts enter on BID and exit on ASK. Same-bar stop/target ambiguity is counted as stop.",
        "US two-year yield is not included because a matching intraday series was not available from the selected BID/ASK data source.",
        "",
        "## Data coverage",
        "",
        pd.DataFrame(data_info).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Top 20 selected only on discovery period",
        "",
        discovery.head(20)[display_columns].to_markdown(index=False, floatfmt=".3f") if not discovery.empty else "No discovery-positive configuration.",
        "",
        "## Validation and forward survivors",
        "",
        f"Count: {len(robust)}",
        "",
        robust[display_columns].to_markdown(index=False, floatfmt=".3f") if not robust.empty else "None passed.",
    ]
    (OUT / "us_momentum_report.md").write_text("\n".join(report), encoding="utf-8")
    print((OUT / "us_momentum_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
