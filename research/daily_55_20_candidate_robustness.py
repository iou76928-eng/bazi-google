from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import diversified_h4_trend_portfolio as core
import diversified_daily_trend_portfolio as daily

OUT = Path("artifacts_daily_candidate")
OUT.mkdir(exist_ok=True)
ASSETS = json.loads(Path("research/portfolio_assets.json").read_text(encoding="utf-8"))["assets"]
CONFIG = core.Config(55, 20, 2.0)
PERIODS = daily.DAILY_PERIODS

CATEGORIES = {
    "metals": ["xauusd", "xagusd", "coppercmdusd"],
    "energy": ["brentcmdusd", "lightcmdusd", "gascmdusd"],
    "agriculture": ["cocoacmdusd", "sugarcmdusd"],
    "bonds": ["ustbondtrusd", "bundtreur"],
    "equities": ["spyususd", "qqqususd"],
    "fx": ["eurusd", "usdjpy"],
    "crypto": ["btcusd"],
}


def select_with_cap(trades, max_positions: int):
    candidates = sorted(trades, key=lambda trade: (pd.Timestamp(trade.entry_time), trade.asset))
    active_exits = []
    accepted = []
    blocked = 0
    for trade in candidates:
        entry = pd.Timestamp(trade.entry_time)
        active_exits = [timestamp for timestamp in active_exits if timestamp > entry]
        if len(active_exits) >= max_positions:
            blocked += 1
            continue
        accepted.append(trade)
        active_exits.append(pd.Timestamp(trade.exit_time))
    return accepted, blocked


def evaluate(trades, cost_r=0.0, risk_fraction=0.0025, max_positions=8):
    accepted, blocked = select_with_cap(trades, max_positions)
    if not accepted:
        return {
            "trades": 0,
            "blocked": blocked,
            "profit_factor": math.nan,
            "avg_r": math.nan,
            "net_r": 0.0,
            "cagr_pct": math.nan,
            "max_dd_pct": math.nan,
            "calmar": math.nan,
            "positive_years": 0,
            "years": 0,
            "worst_year_pct": math.nan,
        }, {}

    r_values = np.array([trade.r_gross - cost_r for trade in accepted], dtype=float)
    gross_profit = float(r_values[r_values > 0].sum())
    gross_loss = float(-r_values[r_values < 0].sum())

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    year_open = {}
    year_close = {}
    for trade in sorted(accepted, key=lambda item: pd.Timestamp(item.exit_time)):
        year = pd.Timestamp(trade.exit_time).year
        year_open.setdefault(year, equity)
        equity *= max(0.000001, 1.0 + risk_fraction * (trade.r_gross - cost_r))
        peak = max(peak, equity)
        max_dd = max(max_dd, 1.0 - equity / peak)
        year_close[year] = equity

    yearly = {year: year_close[year] / year_open[year] - 1.0 for year in year_open}
    first = min(pd.Timestamp(trade.entry_time) for trade in accepted)
    last = max(pd.Timestamp(trade.exit_time) for trade in accepted)
    elapsed = max((last - first).total_seconds() / (365.25 * 86400), 1 / 365.25)
    cagr = equity ** (1.0 / elapsed) - 1.0
    metrics = {
        "trades": len(accepted),
        "blocked": blocked,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else math.inf,
        "avg_r": float(r_values.mean()),
        "net_r": float(r_values.sum()),
        "cagr_pct": cagr * 100.0,
        "max_dd_pct": max_dd * 100.0,
        "calmar": cagr / max_dd if max_dd > 0 else math.inf,
        "positive_years": sum(value > 0 for value in yearly.values()),
        "years": len(yearly),
        "worst_year_pct": min(yearly.values()) * 100.0,
    }
    return metrics, yearly


def subset_years(trades, start_year, end_year):
    return [trade for trade in trades if start_year <= pd.Timestamp(trade.entry_time).year <= end_year]


def add_periods(row, trades, **kwargs):
    for period_name, (start_year, end_year) in PERIODS.items():
        metrics, _ = evaluate(subset_years(trades, start_year, end_year), **kwargs)
        row.update({f"{period_name}_{key}": value for key, value in metrics.items()})
    return row


def main():
    all_trades = []
    starts = {}
    for asset in ASSETS:
        bars = daily.load_daily_asset(asset)
        starts[asset] = bars.index.min().date().isoformat()
        trades = core.simulate_asset(asset, bars, CONFIG)
        all_trades.extend(trades)
        print(asset, len(trades))

    pd.DataFrame([asdict(trade) for trade in all_trades]).to_csv(OUT / "candidate_all_trades.csv", index=False)

    baseline_rows = []
    baseline_yearly = []
    for period_name, (start_year, end_year) in PERIODS.items():
        period_trades = subset_years(all_trades, start_year, end_year)
        metrics, yearly = evaluate(period_trades)
        baseline_rows.append({"period": period_name, **metrics})
        for year, value in yearly.items():
            baseline_yearly.append({"period": period_name, "year": year, "return_pct": value * 100.0})
    pd.DataFrame(baseline_rows).to_csv(OUT / "baseline_periods.csv", index=False)
    pd.DataFrame(baseline_yearly).to_csv(OUT / "baseline_yearly_returns.csv", index=False)

    cost_rows = []
    for cost_r in [0.0, 0.02, 0.05, 0.10, 0.20]:
        row = {"cost_r": cost_r}
        add_periods(row, all_trades, cost_r=cost_r)
        cost_rows.append(row)
    pd.DataFrame(cost_rows).to_csv(OUT / "cost_stress.csv", index=False)

    risk_rows = []
    for risk_fraction in [0.001, 0.0025, 0.005]:
        for max_positions in [4, 8, 15]:
            row = {"risk_pct": risk_fraction * 100.0, "max_positions": max_positions}
            add_periods(row, all_trades, risk_fraction=risk_fraction, max_positions=max_positions)
            risk_rows.append(row)
    pd.DataFrame(risk_rows).to_csv(OUT / "risk_capacity_stress.csv", index=False)

    side_rows = []
    for side_mode in ["BOTH", "LONG", "SHORT"]:
        trades = all_trades if side_mode == "BOTH" else [trade for trade in all_trades if trade.side == side_mode]
        row = {"side_mode": side_mode}
        add_periods(row, trades)
        side_rows.append(row)
    pd.DataFrame(side_rows).to_csv(OUT / "side_decomposition.csv", index=False)

    leave_one_rows = []
    for excluded in ASSETS:
        trades = [trade for trade in all_trades if trade.asset != excluded]
        row = {"excluded_asset": excluded}
        add_periods(row, trades)
        leave_one_rows.append(row)
    pd.DataFrame(leave_one_rows).to_csv(OUT / "leave_one_asset_out.csv", index=False)

    leave_category_rows = []
    for category, excluded_assets in CATEGORIES.items():
        trades = [trade for trade in all_trades if trade.asset not in excluded_assets]
        row = {"excluded_category": category, "excluded_assets": ",".join(excluded_assets)}
        add_periods(row, trades)
        leave_category_rows.append(row)
    pd.DataFrame(leave_category_rows).to_csv(OUT / "leave_category_out.csv", index=False)

    universe_rows = []
    universes = {
        "all_15": ASSETS,
        "no_btc": [asset for asset in ASSETS if asset != "btcusd"],
        "no_crypto_or_agriculture": [asset for asset in ASSETS if asset not in ["btcusd", "cocoacmdusd", "sugarcmdusd"]],
        "legacy_pre2013": [asset for asset, start in starts.items() if start < "2013-01-01"],
        "financial_plus_metals": ["xauusd", "xagusd", "ustbondtrusd", "bundtreur", "spyususd", "qqqususd", "eurusd", "usdjpy", "btcusd"],
    }
    for name, universe in universes.items():
        trades = [trade for trade in all_trades if trade.asset in universe]
        row = {"universe": name, "assets": ",".join(universe), "asset_count": len(universe)}
        add_periods(row, trades)
        universe_rows.append(row)
    pd.DataFrame(universe_rows).to_csv(OUT / "universe_stress.csv", index=False)

    asset_rows = []
    for asset in ASSETS:
        trades = [trade for trade in all_trades if trade.asset == asset]
        row = {"asset": asset, "start": starts[asset]}
        for period_name, (start_year, end_year) in PERIODS.items():
            metrics = core.basic_metrics(subset_years(trades, start_year, end_year))
            row.update({f"{period_name}_{key}": value for key, value in metrics.items()})
        asset_rows.append(row)
    pd.DataFrame(asset_rows).to_csv(OUT / "asset_contributions.csv", index=False)

    baseline = pd.DataFrame(baseline_rows)
    costs = pd.DataFrame(cost_rows)
    sides = pd.DataFrame(side_rows)
    leave_one = pd.DataFrame(leave_one_rows)
    universes_frame = pd.DataFrame(universe_rows)

    validation_survivors = leave_one[
        (leave_one.validation_profit_factor > 1.0)
        & (leave_one.forward_profit_factor > 1.0)
        & (leave_one.validation_avg_r > 0.0)
        & (leave_one.forward_avg_r > 0.0)
    ]
    verdict = (
        len(validation_survivors) >= 12
        and float(costs.loc[costs.cost_r == 0.10, "validation_profit_factor"].iloc[0]) > 1.2
        and float(costs.loc[costs.cost_r == 0.10, "forward_profit_factor"].iloc[0]) > 1.2
        and float(universes_frame.loc[universes_frame.universe == "no_btc", "validation_profit_factor"].iloc[0]) > 1.0
        and float(universes_frame.loc[universes_frame.universe == "no_btc", "forward_profit_factor"].iloc[0]) > 1.0
    )

    report = [
        "# Daily 55/20 Trend Portfolio Robustness",
        "",
        "Fixed candidate: daily 55-day breakout, 20-day opposite-channel exit, ATR20 initial stop at 2 ATR.",
        "Markets: 15 diversified BID/ASK instruments. Baseline portfolio risks 0.25% per trade with a maximum of eight open positions.",
        "Selection history: 2010-2018; validation: 2019-2023; forward: 2024-2026 YTD.",
        "",
        "## Baseline",
        "",
        baseline.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Cost stress",
        "",
        costs.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Long/short decomposition",
        "",
        sides.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Universe stress",
        "",
        universes_frame.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Leave-one-market-out",
        "",
        leave_one.to_markdown(index=False, floatfmt=".3f"),
        "",
        f"Leave-one-out portfolios positive in both validation and forward: {len(validation_survivors)}/{len(leave_one)}.",
        f"Robustness gate: {'PASS' if verdict else 'FAIL'}.",
    ]
    (OUT / "candidate_robustness_report.md").write_text("\n".join(report), encoding="utf-8")
    print((OUT / "candidate_robustness_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
