from __future__ import annotations

import itertools
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from acd_bidask_forward import NY, build_atr, metrics, resample_5m
from acd_bidask_from_csv_v2 import read_side
from ny_2b_bidask_fast import Config, backtest, period_metrics, prepare_contexts

DATA_DIR = Path("data_bidask")
OUT = Path("artifacts_2b")
OUT.mkdir(exist_ok=True)


def main():
    bid = read_side(DATA_DIR / "xauusd_bid_m1.csv", "bid")
    ask = read_side(DATA_DIR / "xauusd_ask_m1.csv", "ask")
    m1 = bid.join(ask, how="inner")
    bars_utc = resample_5m(m1)
    contexts = prepare_contexts(bars_utc.tz_convert(NY), build_atr(bars_utc))

    configs = [
        Config(reference, sweep, confirm, rr, "12:00", side, trend, 0.0)
        for reference, sweep, confirm, rr, side, trend in itertools.product(
            ["prior_day", "overnight", "premarket"],
            [0.00, 0.03, 0.05],
            ["same_bar", "within_3"],
            [1.0, 1.5],
            ["both", "long_only", "short_only"],
            [False, True],
        )
    ]

    rows = []
    cache = {}
    for cfg in configs:
        trades = backtest(contexts, cfg)
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

    columns = [
        "reference", "sweep_atr", "confirm_mode", "rr", "side_mode", "trend_filter",
        "y2024_trades", "y2024_win_rate", "y2024_profit_factor", "y2024_avg_r", "y2024_max_dd_r",
        "y2025_trades", "y2025_win_rate", "y2025_profit_factor", "y2025_avg_r", "y2025_max_dd_r",
        "y2026_trades", "y2026_win_rate", "y2026_profit_factor", "y2026_avg_r", "y2026_max_dd_r",
        "all_trades", "all_profit_factor", "all_avg_r", "all_max_dd_r",
    ]
    report = [
        "# NY Open 2B True BID/ASK Quick Test",
        "",
        "Research adaptation of Sperandeo 2B; 12:00 forced exit, no stop buffer, extra cost 0.05.",
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
