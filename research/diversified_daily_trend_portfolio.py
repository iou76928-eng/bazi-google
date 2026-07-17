from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import diversified_h4_trend_portfolio as core

OUT = Path("artifacts_daily_portfolio")
OUT.mkdir(exist_ok=True)
ASSET_FILE = Path("research/portfolio_assets.json")

DAILY_PERIODS = {
    "discovery": (2010, 2018),
    "validation": (2019, 2023),
    "forward": (2024, 2026),
    "all": (2010, 2026),
}


def load_daily_asset(asset: str) -> pd.DataFrame:
    h4 = core.read_side(asset, "bid").join(core.read_side(asset, "ask"), how="inner")
    h4 = h4[(h4.index >= core.START) & (h4.index < core.END)]
    aggregations = {
        "bid_open": "first", "bid_high": "max", "bid_low": "min", "bid_close": "last",
        "ask_open": "first", "ask_high": "max", "ask_low": "min", "ask_close": "last",
    }
    bars = h4.resample("1D", label="left", closed="left").agg(aggregations)
    bars = bars.dropna(subset=list(aggregations))
    if len(bars) < 300:
        raise RuntimeError(f"{asset}: insufficient daily data ({len(bars)} rows)")
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
    bars["atr"] = true_range.ewm(alpha=1.0 / 20.0, adjust=False, min_periods=20).mean()
    return bars


def main():
    assets = json.loads(ASSET_FILE.read_text(encoding="utf-8"))["assets"]
    configs = [
        core.Config(20, 10, 2.0),
        core.Config(20, 10, 2.5),
        core.Config(55, 20, 2.0),
        core.Config(55, 20, 2.5),
        core.Config(100, 40, 2.0),
        core.Config(100, 40, 2.5),
        core.Config(150, 60, 2.5),
    ]

    all_by_config = {config.name: [] for config in configs}
    data_rows = []
    for asset in assets:
        bars = load_daily_asset(asset)
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
            all_by_config[config.name].extend(core.simulate_asset(asset, bars, config))
        print(f"{asset}: {len(bars):,} daily bars")

    result_rows = []
    asset_rows = []
    for config in configs:
        config_trades = all_by_config[config.name]
        row = asdict(config)
        row["config"] = config.name
        for period_name, (start_year, end_year) in DAILY_PERIODS.items():
            subset = core.period_subset(config_trades, start_year, end_year)
            portfolio = core.portfolio_metrics(subset, cost_r=0.0)
            positive_assets, asset_frame = core.asset_breadth(subset, assets, cost_r=0.0)
            row.update({f"{period_name}_{key}": value for key, value in portfolio.items()})
            row[f"{period_name}_positive_assets"] = positive_assets
            for record in asset_frame.to_dict("records"):
                asset_rows.append({"config": config.name, "period": period_name, **record})
        result_rows.append(row)

    results = pd.DataFrame(result_rows)
    results["discovery_score"] = (
        results.discovery_cagr_pct
        + 3.0 * results.discovery_avg_r
        - 0.25 * results.discovery_max_dd_pct
    )
    discovery = results[
        (results.discovery_trades >= 150)
        & (results.discovery_profit_factor > 1.0)
        & (results.discovery_avg_r > 0.0)
        & (results.discovery_cagr_pct > 0.0)
    ].sort_values(["discovery_score", "discovery_profit_factor"], ascending=False)

    robust = discovery[
        (discovery.validation_trades >= 100)
        & (discovery.forward_trades >= 50)
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

    results.to_csv(OUT / "daily_portfolio_grid.csv", index=False)
    discovery.to_csv(OUT / "daily_portfolio_discovery_ranked.csv", index=False)
    robust.to_csv(OUT / "daily_portfolio_robust_candidates.csv", index=False)
    pd.DataFrame(asset_rows).to_csv(OUT / "daily_portfolio_asset_results.csv", index=False)
    pd.DataFrame(data_rows).to_csv(OUT / "daily_portfolio_data_info.csv", index=False)

    cost_rows = []
    winner_rows = []
    if not discovery.empty:
        winner_name = str(discovery.iloc[0].config)
        winner_trades = all_by_config[winner_name]
        for cost_r in [0.0, 0.02, 0.05, 0.10]:
            row = {"config": winner_name, "cost_r": cost_r}
            for period_name, (start_year, end_year) in DAILY_PERIODS.items():
                subset = core.period_subset(winner_trades, start_year, end_year)
                portfolio = core.portfolio_metrics(subset, cost_r=cost_r)
                positive_assets, _ = core.asset_breadth(subset, assets, cost_r=cost_r)
                row.update({f"{period_name}_{key}": value for key, value in portfolio.items()})
                row[f"{period_name}_positive_assets"] = positive_assets
            cost_rows.append(row)
        accepted, _, _ = core.select_with_position_cap(winner_trades)
        winner_rows = [asdict(trade) for trade in accepted]

    pd.DataFrame(cost_rows).to_csv(OUT / "daily_portfolio_cost_stress.csv", index=False)
    pd.DataFrame(winner_rows).to_csv(OUT / "daily_portfolio_winner_trades.csv", index=False)

    display_columns = [
        "config",
        "discovery_trades", "discovery_profit_factor", "discovery_avg_r", "discovery_cagr_pct", "discovery_max_dd_pct", "discovery_positive_assets",
        "validation_trades", "validation_profit_factor", "validation_avg_r", "validation_cagr_pct", "validation_max_dd_pct", "validation_positive_assets",
        "forward_trades", "forward_profit_factor", "forward_avg_r", "forward_cagr_pct", "forward_max_dd_pct", "forward_positive_assets",
        "all_trades", "all_profit_factor", "all_avg_r", "all_cagr_pct", "all_max_dd_pct", "all_calmar", "all_positive_assets",
    ]
    report = [
        "# Diversified Daily Trend Portfolio Validation",
        "",
        f"Markets: {', '.join(assets)}.",
        "Data: Dukascopy BID/ASK H4 aggregated to UTC daily bars, 2010-01-01 through 2026-07-16 subject to availability.",
        "Rules: same daily Donchian parameters on every market; ATR20 initial stop; opposite channel exit; no EMA and no fixed target.",
        "Portfolio: 0.25% equity risk per accepted trade, maximum eight simultaneous positions.",
        "Selection: 2010-2018. Validation: 2019-2023. Forward: 2024-2026 YTD.",
        "",
        "## Data coverage",
        "",
        pd.DataFrame(data_rows).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Discovery-ranked configurations",
        "",
        discovery[display_columns].to_markdown(index=False, floatfmt=".3f") if not discovery.empty else "No discovery-positive configuration.",
        "",
        "## Robust daily portfolio candidates",
        "",
        f"Count: {len(robust)}",
        "",
        robust[display_columns].to_markdown(index=False, floatfmt=".3f") if not robust.empty else "None passed.",
        "",
        "## Discovery winner cost stress",
        "",
        pd.DataFrame(cost_rows).to_markdown(index=False, floatfmt=".3f") if cost_rows else "No discovery winner.",
    ]
    (OUT / "daily_portfolio_report.md").write_text("\n".join(report), encoding="utf-8")
    print((OUT / "daily_portfolio_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
