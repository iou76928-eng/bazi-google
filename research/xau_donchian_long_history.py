from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data_xau_h4")
OUT = Path("artifacts_xau_long")
OUT.mkdir(exist_ok=True)


@dataclass
class Trade:
    config: str
    side: str
    entry_time: str
    exit_time: str
    entry: float
    exit: float
    risk: float
    gross_points: float
    r_net: float


def read_side(side: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA / f"xauusd_{side}_h4.csv")
    index = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    prefix = side
    result = pd.DataFrame(
        {
            f"{prefix}_open": pd.to_numeric(frame["open"], errors="coerce").to_numpy(),
            f"{prefix}_high": pd.to_numeric(frame["high"], errors="coerce").to_numpy(),
            f"{prefix}_low": pd.to_numeric(frame["low"], errors="coerce").to_numpy(),
            f"{prefix}_close": pd.to_numeric(frame["close"], errors="coerce").to_numpy(),
        },
        index=pd.DatetimeIndex(index),
    ).dropna().sort_index()
    return result[~result.index.duplicated(keep="last")]


def load_bars() -> pd.DataFrame:
    bars = read_side("bid").join(read_side("ask"), how="inner")
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
    bars["atr"] = true_range.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    return bars


def simulate(bars: pd.DataFrame, entry_length: int, exit_length: int, stop_atr: float, extra_cost: float = 0.0):
    index = bars.index
    bid_open = bars.bid_open.to_numpy(float)
    bid_low = bars.bid_low.to_numpy(float)
    bid_close = bars.bid_close.to_numpy(float)
    ask_open = bars.ask_open.to_numpy(float)
    ask_high = bars.ask_high.to_numpy(float)
    ask_close = bars.ask_close.to_numpy(float)
    atr = bars.atr.to_numpy(float)

    entry_high = bars.mid_high.rolling(entry_length, min_periods=entry_length).max().shift(1).to_numpy(float)
    entry_low = bars.mid_low.rolling(entry_length, min_periods=entry_length).min().shift(1).to_numpy(float)
    exit_low = bars.mid_low.rolling(exit_length, min_periods=exit_length).min().shift(1).to_numpy(float)
    exit_high = bars.mid_high.rolling(exit_length, min_periods=exit_length).max().shift(1).to_numpy(float)

    config = f"N={entry_length}|M={exit_length}|stopATR={stop_atr:.2f}|cost={extra_cost:.2f}"
    trades = []
    bar = max(100, entry_length)
    while bar < len(bars) - 2:
        if not np.isfinite(atr[bar]) or not np.isfinite(entry_high[bar]) or not np.isfinite(entry_low[bar]):
            bar += 1
            continue
        long_hit = ask_high[bar] >= entry_high[bar]
        short_hit = bid_low[bar] <= entry_low[bar]
        if long_hit == short_hit:
            bar += 1
            continue

        direction = 1 if long_hit else -1
        entry = max(float(entry_high[bar]), float(ask_open[bar])) if direction > 0 else min(float(entry_low[bar]), float(bid_open[bar]))
        initial_stop = entry - direction * stop_atr * atr[bar]
        risk = abs(entry - initial_stop)
        if not np.isfinite(risk) or risk <= 0:
            bar += 1
            continue

        position_bar = bar
        active_stop = initial_stop
        exit_price = None
        while position_bar < len(bars):
            if direction > 0:
                if np.isfinite(exit_low[position_bar]):
                    active_stop = max(initial_stop, float(exit_low[position_bar]))
                if bid_low[position_bar] <= active_stop:
                    exit_price = min(active_stop, float(bid_open[position_bar]))
                    break
            else:
                if np.isfinite(exit_high[position_bar]):
                    active_stop = min(initial_stop, float(exit_high[position_bar]))
                if ask_high[position_bar] >= active_stop:
                    exit_price = max(active_stop, float(ask_open[position_bar]))
                    break
            position_bar += 1

        if exit_price is None:
            position_bar = len(bars) - 1
            exit_price = float(bid_close[position_bar] if direction > 0 else ask_close[position_bar])

        gross = direction * (exit_price - entry)
        r_net = (gross - extra_cost) / risk
        trades.append(
            Trade(
                config=config,
                side="LONG" if direction > 0 else "SHORT",
                entry_time=index[bar].isoformat(),
                exit_time=index[position_bar].isoformat(),
                entry=float(entry),
                exit=float(exit_price),
                risk=float(risk),
                gross_points=float(gross),
                r_net=float(r_net),
            )
        )
        bar = max(bar + 1, position_bar + 1)
    return trades


def metrics(trades):
    if not trades:
        return {"trades": 0, "win_rate": math.nan, "profit_factor": math.nan, "net_r": 0.0, "avg_r": math.nan, "max_dd_r": math.nan}
    ordered = sorted(trades, key=lambda trade: trade.entry_time)
    values = np.array([trade.r_net for trade in ordered], dtype=float)
    gross_profit = values[values > 0].sum()
    gross_loss = -values[values < 0].sum()
    equity = np.cumsum(values)
    peaks = np.maximum.accumulate(np.r_[0.0, equity])
    drawdowns = peaks[1:] - equity
    return {
        "trades": int(len(values)),
        "win_rate": float((values > 0).mean() * 100.0),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else math.inf,
        "net_r": float(values.sum()),
        "avg_r": float(values.mean()),
        "max_dd_r": float(drawdowns.max()) if len(drawdowns) else 0.0,
    }


def subset(trades, start_year: int, end_year: int):
    return [trade for trade in trades if start_year <= pd.Timestamp(trade.entry_time).year <= end_year]


def main():
    bars = load_bars()
    configurations = [
        (20, 10, 2.0),
        (20, 10, 2.25),
        (24, 12, 2.0),
        (24, 12, 2.25),
        (30, 15, 2.0),
        (30, 15, 2.25),
    ]
    rows = []
    trade_cache = {}
    for entry_length, exit_length, stop_atr in configurations:
        trades = simulate(bars, entry_length, exit_length, stop_atr, 0.0)
        trade_cache[(entry_length, exit_length, stop_atr)] = trades
        row = {"N": entry_length, "M": exit_length, "stopATR": stop_atr}
        row.update({f"all_{key}": value for key, value in metrics(trades).items()})
        for year in range(2010, 2027):
            row.update({f"y{year}_{key}": value for key, value in metrics(subset(trades, year, year)).items()})
        for name, start_year, end_year in [
            ("b2010_2014", 2010, 2014),
            ("b2015_2019", 2015, 2019),
            ("b2020_2023", 2020, 2023),
            ("b2024_2026", 2024, 2026),
        ]:
            row.update({f"{name}_{key}": value for key, value in metrics(subset(trades, start_year, end_year)).items()})
        rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "long_history_grid.csv", index=False)

    primary = trade_cache[(24, 12, 2.0)]
    pd.DataFrame([asdict(trade) for trade in primary]).to_csv(OUT / "primary_trades.csv", index=False)

    yearly_rows = []
    for year in range(2010, 2027):
        year_trades = subset(primary, year, year)
        row = {"year": year}
        row.update(metrics(year_trades))
        row.update({f"long_{key}": value for key, value in metrics([trade for trade in year_trades if trade.side == "LONG"]).items()})
        row.update({f"short_{key}": value for key, value in metrics([trade for trade in year_trades if trade.side == "SHORT"]).items()})
        yearly_rows.append(row)
    yearly = pd.DataFrame(yearly_rows)
    yearly.to_csv(OUT / "primary_yearly.csv", index=False)

    stress_rows = []
    for cost in [0.0, 0.25, 0.50, 1.00]:
        trades = simulate(bars, 24, 12, 2.0, cost)
        row = {"extra_cost": cost}
        row.update({f"all_{key}": value for key, value in metrics(trades).items()})
        for name, start_year, end_year in [
            ("b2010_2014", 2010, 2014),
            ("b2015_2019", 2015, 2019),
            ("b2020_2023", 2020, 2023),
            ("b2024_2026", 2024, 2026),
        ]:
            row.update({f"{name}_{key}": value for key, value in metrics(subset(trades, start_year, end_year)).items()})
        stress_rows.append(row)
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(OUT / "primary_cost_stress.csv", index=False)

    block_columns = [
        "N", "M", "stopATR",
        "b2010_2014_trades", "b2010_2014_profit_factor", "b2010_2014_avg_r", "b2010_2014_max_dd_r",
        "b2015_2019_trades", "b2015_2019_profit_factor", "b2015_2019_avg_r", "b2015_2019_max_dd_r",
        "b2020_2023_trades", "b2020_2023_profit_factor", "b2020_2023_avg_r", "b2020_2023_max_dd_r",
        "b2024_2026_trades", "b2024_2026_profit_factor", "b2024_2026_avg_r", "b2024_2026_max_dd_r",
        "all_trades", "all_profit_factor", "all_avg_r", "all_max_dd_r",
    ]
    yearly_columns = ["year", "trades", "win_rate", "profit_factor", "avg_r", "max_dd_r", "long_trades", "long_profit_factor", "long_avg_r", "short_trades", "short_profit_factor", "short_avg_r"]
    report = [
        "# XAUUSD 4H Donchian Long-History Validation",
        "",
        "Data: Dukascopy XAUUSD BID/ASK H4, 2010-01-01 through 2026-07-16.",
        "Primary parameters were fixed before this test: entry 24 bars, exit 12 bars, initial stop 2 ATR. No EMA and no fixed profit target.",
        "The 2010-2023 period was not used to select the primary parameters and is the key regime check.",
        "",
        "## Parameter neighborhood by market block",
        "",
        results[block_columns].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Primary candidate by year",
        "",
        yearly[yearly_columns].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Primary candidate cost stress",
        "",
        stress.to_markdown(index=False, floatfmt=".3f"),
    ]
    (OUT / "long_history_report.md").write_text("\n".join(report), encoding="utf-8")
    print((OUT / "long_history_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
