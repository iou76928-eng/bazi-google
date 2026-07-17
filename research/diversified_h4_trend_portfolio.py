from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data_portfolio")
OUT = Path("artifacts_portfolio")
OUT.mkdir(exist_ok=True)
ASSET_FILE = Path("research/portfolio_assets.json")

START = pd.Timestamp("2010-01-01", tz="UTC")
END = pd.Timestamp("2026-07-17", tz="UTC")
RISK_PER_TRADE = 0.0025
MAX_POSITIONS = 8

PERIODS = {
    "discovery": (2010, 2018),
    "validation": (2019, 2023),
    "forward": (2024, 2026),
    "all": (2010, 2026),
}


@dataclass(frozen=True)
class Config:
    entry_length: int
    exit_length: int
    stop_atr: float

    @property
    def name(self) -> str:
        return f"N={self.entry_length}|M={self.exit_length}|stopATR={self.stop_atr:.2f}"


@dataclass
class Trade:
    asset: str
    config: str
    side: str
    entry_time: str
    exit_time: str
    entry: float
    exit: float
    risk_price: float
    r_gross: float


def read_side(asset: str, side: str) -> pd.DataFrame:
    path = DATA_DIR / f"{asset}_{side}_h4.csv"
    frame = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
    index = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    result = pd.DataFrame(
        {
            f"{side}_open": pd.to_numeric(frame["open"], errors="coerce").to_numpy(),
            f"{side}_high": pd.to_numeric(frame["high"], errors="coerce").to_numpy(),
            f"{side}_low": pd.to_numeric(frame["low"], errors="coerce").to_numpy(),
            f"{side}_close": pd.to_numeric(frame["close"], errors="coerce").to_numpy(),
        },
        index=pd.DatetimeIndex(index),
    ).dropna().sort_index()
    return result[~result.index.duplicated(keep="last")]


def load_asset(asset: str) -> pd.DataFrame:
    bars = read_side(asset, "bid").join(read_side(asset, "ask"), how="inner")
    bars = bars[(bars.index >= START) & (bars.index < END)]
    if len(bars) < 500:
        raise RuntimeError(f"{asset}: insufficient merged H4 data ({len(bars)} rows)")
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


def simulate_asset(asset: str, bars: pd.DataFrame, config: Config) -> list[Trade]:
    index = bars.index
    bid_open = bars.bid_open.to_numpy(float)
    bid_low = bars.bid_low.to_numpy(float)
    bid_close = bars.bid_close.to_numpy(float)
    ask_open = bars.ask_open.to_numpy(float)
    ask_high = bars.ask_high.to_numpy(float)
    ask_close = bars.ask_close.to_numpy(float)
    atr = bars.atr.to_numpy(float)

    entry_high = bars.mid_high.rolling(config.entry_length, min_periods=config.entry_length).max().shift(1).to_numpy(float)
    entry_low = bars.mid_low.rolling(config.entry_length, min_periods=config.entry_length).min().shift(1).to_numpy(float)
    exit_low = bars.mid_low.rolling(config.exit_length, min_periods=config.exit_length).min().shift(1).to_numpy(float)
    exit_high = bars.mid_high.rolling(config.exit_length, min_periods=config.exit_length).max().shift(1).to_numpy(float)

    trades: list[Trade] = []
    bar = max(100, config.entry_length)
    while bar < len(bars) - 1:
        if not np.isfinite(atr[bar]) or not np.isfinite(entry_high[bar]) or not np.isfinite(entry_low[bar]):
            bar += 1
            continue

        long_hit = ask_high[bar] >= entry_high[bar]
        short_hit = bid_low[bar] <= entry_low[bar]
        if long_hit == short_hit:
            bar += 1
            continue

        direction = 1 if long_hit else -1
        entry = (
            max(float(entry_high[bar]), float(ask_open[bar]))
            if direction > 0
            else min(float(entry_low[bar]), float(bid_open[bar]))
        )
        initial_stop = entry - direction * config.stop_atr * atr[bar]
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

        gross_points = direction * (exit_price - entry)
        trades.append(
            Trade(
                asset=asset,
                config=config.name,
                side="LONG" if direction > 0 else "SHORT",
                entry_time=index[bar].isoformat(),
                exit_time=index[position_bar].isoformat(),
                entry=float(entry),
                exit=float(exit_price),
                risk_price=float(risk),
                r_gross=float(gross_points / risk),
            )
        )
        bar = max(bar + 1, position_bar + 1)
    return trades


def trade_year(trade: Trade) -> int:
    return pd.Timestamp(trade.entry_time).year


def period_subset(trades: list[Trade], start_year: int, end_year: int) -> list[Trade]:
    return [trade for trade in trades if start_year <= trade_year(trade) <= end_year]


def select_with_position_cap(trades: list[Trade], max_positions: int = MAX_POSITIONS):
    candidates = sorted(trades, key=lambda trade: (pd.Timestamp(trade.entry_time), trade.asset))
    active_exits: list[pd.Timestamp] = []
    accepted: list[Trade] = []
    blocked = 0
    max_raw_overlap = 0

    raw_events = []
    for trade in candidates:
        raw_events.append((pd.Timestamp(trade.entry_time), 1))
        raw_events.append((pd.Timestamp(trade.exit_time), -1))
    running = 0
    for timestamp, delta in sorted(raw_events, key=lambda item: (item[0], item[1])):
        running += delta
        max_raw_overlap = max(max_raw_overlap, running)

    for trade in candidates:
        entry_time = pd.Timestamp(trade.entry_time)
        active_exits = [exit_time for exit_time in active_exits if exit_time > entry_time]
        if len(active_exits) >= max_positions:
            blocked += 1
            continue
        accepted.append(trade)
        active_exits.append(pd.Timestamp(trade.exit_time))
    return accepted, blocked, max_raw_overlap


def basic_metrics(trades: list[Trade], cost_r: float = 0.0):
    if not trades:
        return {
            "trades": 0,
            "win_rate": math.nan,
            "profit_factor": math.nan,
            "net_r": 0.0,
            "avg_r": math.nan,
            "max_dd_r": math.nan,
        }
    ordered = sorted(trades, key=lambda trade: trade.exit_time)
    values = np.array([trade.r_gross - cost_r for trade in ordered], dtype=float)
    gross_profit = float(values[values > 0].sum())
    gross_loss = float(-values[values < 0].sum())
    equity_r = np.cumsum(values)
    peaks = np.maximum.accumulate(np.r_[0.0, equity_r])
    drawdowns = peaks[1:] - equity_r
    return {
        "trades": int(len(values)),
        "win_rate": float((values > 0).mean() * 100.0),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else math.inf,
        "net_r": float(values.sum()),
        "avg_r": float(values.mean()),
        "max_dd_r": float(drawdowns.max()) if len(drawdowns) else 0.0,
    }


def portfolio_metrics(trades: list[Trade], cost_r: float = 0.0):
    accepted, blocked, max_raw_overlap = select_with_position_cap(trades)
    base = basic_metrics(accepted, cost_r)
    if not accepted:
        return {
            **base,
            "blocked": blocked,
            "max_raw_overlap": max_raw_overlap,
            "total_return_pct": 0.0,
            "cagr_pct": math.nan,
            "max_dd_pct": math.nan,
            "calmar": math.nan,
            "positive_years": 0,
            "years": 0,
            "worst_year_pct": math.nan,
        }

    events = sorted(accepted, key=lambda trade: pd.Timestamp(trade.exit_time))
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    yearly_start: dict[int, float] = {}
    yearly_end: dict[int, float] = {}

    for trade in events:
        year = pd.Timestamp(trade.exit_time).year
        yearly_start.setdefault(year, equity)
        r_value = trade.r_gross - cost_r
        equity *= max(0.000001, 1.0 + RISK_PER_TRADE * r_value)
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, 1.0 - equity / peak)
        yearly_end[year] = equity

    yearly_returns = {
        year: yearly_end[year] / yearly_start[year] - 1.0
        for year in yearly_start
        if year in yearly_end
    }
    first = pd.Timestamp(min(trade.entry_time for trade in accepted))
    last = pd.Timestamp(max(trade.exit_time for trade in accepted))
    years_elapsed = max((last - first).total_seconds() / (365.25 * 24 * 3600), 1 / 365.25)
    cagr = equity ** (1.0 / years_elapsed) - 1.0
    calmar = cagr / max_drawdown if max_drawdown > 0 else math.inf

    return {
        **base,
        "blocked": int(blocked),
        "max_raw_overlap": int(max_raw_overlap),
        "total_return_pct": float((equity - 1.0) * 100.0),
        "cagr_pct": float(cagr * 100.0),
        "max_dd_pct": float(max_drawdown * 100.0),
        "calmar": float(calmar),
        "positive_years": int(sum(value > 0 for value in yearly_returns.values())),
        "years": int(len(yearly_returns)),
        "worst_year_pct": float(min(yearly_returns.values()) * 100.0) if yearly_returns else math.nan,
    }


def asset_breadth(trades: list[Trade], assets: list[str], cost_r: float = 0.0):
    positive = 0
    rows = []
    for asset in assets:
        asset_trades = [trade for trade in trades if trade.asset == asset]
        result = basic_metrics(asset_trades, cost_r)
        if result["trades"] >= 10 and result["profit_factor"] > 1.0 and result["avg_r"] > 0:
            positive += 1
        rows.append({"asset": asset, **result})
    return positive, pd.DataFrame(rows)


def main():
    assets = json.loads(ASSET_FILE.read_text(encoding="utf-8"))["assets"]
    configs = [
        Config(20, 10, 2.0),
        Config(20, 10, 2.25),
        Config(24, 12, 2.0),
        Config(24, 12, 2.25),
        Config(30, 15, 2.0),
        Config(30, 15, 2.25),
    ]

    all_by_config: dict[str, list[Trade]] = {config.name: [] for config in configs}
    data_rows = []
    for asset in assets:
        bars = load_asset(asset)
        data_rows.append(
            {
                "asset": asset,
                "rows": len(bars),
                "start": bars.index.min().isoformat(),
                "end": bars.index.max().isoformat(),
                "median_spread": float((bars.ask_open - bars.bid_open).median()),
            }
        )
        for config in configs:
            trades = simulate_asset(asset, bars, config)
            all_by_config[config.name].extend(trades)
        print(f"{asset}: {len(bars):,} H4 bars")

    result_rows = []
    asset_rows = []
    for config in configs:
        config_trades = all_by_config[config.name]
        row = asdict(config)
        row["config"] = config.name
        for period_name, (start_year, end_year) in PERIODS.items():
            subset = period_subset(config_trades, start_year, end_year)
            portfolio = portfolio_metrics(subset, cost_r=0.0)
            positive_assets, asset_frame = asset_breadth(subset, assets, cost_r=0.0)
            row.update({f"{period_name}_{key}": value for key, value in portfolio.items()})
            row[f"{period_name}_positive_assets"] = positive_assets
            for record in asset_frame.to_dict("records"):
                asset_rows.append(
                    {
                        "config": config.name,
                        "period": period_name,
                        **record,
                    }
                )
        result_rows.append(row)

    results = pd.DataFrame(result_rows)
    results["discovery_score"] = (
        results.discovery_cagr_pct
        + 2.0 * results.discovery_avg_r
        - 0.25 * results.discovery_max_dd_pct
    )
    discovery = results[
        (results.discovery_trades >= 500)
        & (results.discovery_profit_factor > 1.0)
        & (results.discovery_avg_r > 0.0)
        & (results.discovery_cagr_pct > 0.0)
    ].sort_values(["discovery_score", "discovery_profit_factor"], ascending=False)

    robust = discovery[
        (discovery.validation_trades >= 300)
        & (discovery.forward_trades >= 150)
        & (discovery.validation_profit_factor > 1.05)
        & (discovery.forward_profit_factor > 1.05)
        & (discovery.validation_avg_r > 0.0)
        & (discovery.forward_avg_r > 0.0)
        & (discovery.validation_cagr_pct > 0.0)
        & (discovery.forward_cagr_pct > 0.0)
        & (discovery.validation_positive_assets >= 5)
        & (discovery.forward_positive_assets >= 5)
        & (discovery.all_max_dd_pct <= 25.0)
    ].copy()

    results.to_csv(OUT / "portfolio_grid.csv", index=False)
    discovery.to_csv(OUT / "portfolio_discovery_ranked.csv", index=False)
    robust.to_csv(OUT / "portfolio_robust_candidates.csv", index=False)
    pd.DataFrame(asset_rows).to_csv(OUT / "portfolio_asset_results.csv", index=False)
    pd.DataFrame(data_rows).to_csv(OUT / "portfolio_data_info.csv", index=False)

    cost_rows = []
    accepted_rows = []
    if not discovery.empty:
        winner_name = str(discovery.iloc[0].config)
        winner_trades = all_by_config[winner_name]
        for cost_r in [0.0, 0.02, 0.05, 0.10]:
            cost_row = {"config": winner_name, "cost_r": cost_r}
            for period_name, (start_year, end_year) in PERIODS.items():
                subset = period_subset(winner_trades, start_year, end_year)
                portfolio = portfolio_metrics(subset, cost_r=cost_r)
                positive_assets, _ = asset_breadth(subset, assets, cost_r=cost_r)
                cost_row.update({f"{period_name}_{key}": value for key, value in portfolio.items()})
                cost_row[f"{period_name}_positive_assets"] = positive_assets
            cost_rows.append(cost_row)
        accepted, _, _ = select_with_position_cap(winner_trades)
        accepted_rows = [asdict(trade) for trade in accepted]

    pd.DataFrame(cost_rows).to_csv(OUT / "portfolio_cost_stress.csv", index=False)
    pd.DataFrame(accepted_rows).to_csv(OUT / "portfolio_winner_trades.csv", index=False)

    display_columns = [
        "config",
        "discovery_trades", "discovery_profit_factor", "discovery_avg_r", "discovery_cagr_pct", "discovery_max_dd_pct", "discovery_positive_assets",
        "validation_trades", "validation_profit_factor", "validation_avg_r", "validation_cagr_pct", "validation_max_dd_pct", "validation_positive_assets",
        "forward_trades", "forward_profit_factor", "forward_avg_r", "forward_cagr_pct", "forward_max_dd_pct", "forward_positive_assets",
        "all_trades", "all_profit_factor", "all_avg_r", "all_cagr_pct", "all_max_dd_pct", "all_calmar", "all_positive_assets",
    ]
    report = [
        "# Diversified H4 Trend Portfolio Validation",
        "",
        f"Markets: {', '.join(assets)}.",
        "Data: Dukascopy BID/ASK H4, 2010-01-01 through 2026-07-16, subject to each market's available history.",
        "Rules: same Donchian parameters on every market; ATR14 initial stop; opposite channel exit; no EMA and no fixed profit target.",
        "Portfolio: 0.25% equity risk per accepted trade, maximum eight simultaneous positions (2% initial-risk cap).",
        "Selection: 2010-2018. Validation: 2019-2023. Forward: 2024-2026 YTD.",
        "Position-cap filtering uses independently generated per-market trades and is intentionally conservative when a blocked trade would have freed the market for a later signal.",
        "",
        "## Data coverage",
        "",
        pd.DataFrame(data_rows).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Discovery-ranked parameter neighborhood",
        "",
        discovery[display_columns].to_markdown(index=False, floatfmt=".3f") if not discovery.empty else "No discovery-positive configuration.",
        "",
        "## Robust portfolio candidates",
        "",
        f"Count: {len(robust)}",
        "",
        robust[display_columns].to_markdown(index=False, floatfmt=".3f") if not robust.empty else "None passed.",
        "",
        "## Discovery winner cost stress",
        "",
        pd.DataFrame(cost_rows).to_markdown(index=False, floatfmt=".3f") if cost_rows else "No discovery winner.",
    ]
    (OUT / "portfolio_report.md").write_text("\n".join(report), encoding="utf-8")
    print((OUT / "portfolio_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
