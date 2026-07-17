from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from acd_bidask_forward import (
    Config,
    END_TEST,
    NY,
    START_TEST,
    backtest,
    build_atr,
    metrics,
    period_metrics,
    resample_5m,
)

DATA_DIR = Path("data_bidask")
OUT = Path("artifacts_bidask")
OUT.mkdir(exist_ok=True)


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
            f"{prefix}_open": pd.to_numeric(frame["open"], errors="coerce"),
            f"{prefix}_high": pd.to_numeric(frame["high"], errors="coerce"),
            f"{prefix}_low": pd.to_numeric(frame["low"], errors="coerce"),
            f"{prefix}_close": pd.to_numeric(frame["close"], errors="coerce"),
            f"{prefix}_volume": pd.to_numeric(frame["volume"], errors="coerce"),
        },
        index=pd.DatetimeIndex(index),
    )
    result = result.dropna().sort_index()
    return result[~result.index.duplicated(keep="last")]


def main() -> None:
    bid = read_side(DATA_DIR / "xauusd_bid_m1.csv", "bid")
    ask = read_side(DATA_DIR / "xauusd_ask_m1.csv", "ask")
    m1 = bid.join(ask, how="inner")
    if len(m1) < 500_000:
        raise RuntimeError(f"Insufficient merged BID/ASK rows: {len(m1)}")

    bars_utc = resample_5m(m1)
    atr = build_atr(bars_utc)
    bars_ny = bars_utc.tz_convert(NY)

    median_price = float(bars_utc.mid_close.median())
    median_spread = float(bars_utc.spread_open.median())
    p90_spread = float(bars_utc.spread_open.quantile(0.90))
    if not 500 <= median_price <= 10000:
        raise RuntimeError(f"Invalid median price: {median_price}")
    if not 0 < median_spread < 20:
        raise RuntimeError(f"Invalid median spread: {median_spread}")

    configs = [
        Config(a_mult, entry_mode, rr, extra_cost)
        for a_mult, entry_mode, rr, extra_cost in itertools.product(
            [0.03, 0.05, 0.08],
            ["touch", "close1"],
            [0.50, 0.75, 1.00],
            [0.00, 0.05, 0.10, 0.25],
        )
    ]

    rows = []
    cache = {}
    for config in configs:
        trades = backtest(bars_ny, atr, config)
        cache[config] = trades
        row = asdict(config)
        row.update({f"all_{key}": value for key, value in metrics(trades).items()})
        for year in [2024, 2025, 2026]:
            row.update({f"y{year}_{key}": value for key, value in period_metrics(trades, year).items()})
        if trades:
            spreads = [trade.entry_spread for trade in trades]
            row["median_entry_spread"] = float(np.median(spreads))
            row["p90_entry_spread"] = float(np.quantile(spreads, 0.90))
        else:
            row["median_entry_spread"] = math.nan
            row["p90_entry_spread"] = math.nan
        rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "bidask_grid.csv", index=False)

    discovery = results[
        (results.y2024_trades >= 25)
        & np.isfinite(results.y2024_profit_factor)
        & (results.y2024_profit_factor > 1.0)
        & (results.y2024_avg_r > 0.0)
    ].copy()
    discovery["discovery_score"] = (
        discovery.y2024_avg_r.clip(lower=-1, upper=2) * np.sqrt(discovery.y2024_trades)
        - 0.025 * discovery.y2024_max_dd_r
    )
    discovery = discovery.sort_values(["discovery_score", "y2024_profit_factor"], ascending=False)
    discovery.to_csv(OUT / "bidask_discovery_ranked.csv", index=False)

    robust = discovery[
        (discovery.y2025_trades >= 25)
        & (discovery.y2026_trades >= 10)
        & (discovery.y2025_profit_factor > 1.0)
        & (discovery.y2026_profit_factor > 1.0)
        & (discovery.y2025_avg_r > 0.0)
        & (discovery.y2026_avg_r > 0.0)
    ].copy()
    robust.to_csv(OUT / "bidask_forward_positive.csv", index=False)

    if not discovery.empty:
        best = discovery.iloc[0]
        best_config = Config(float(best.a_mult), str(best.entry_mode), float(best.rr), float(best.extra_cost))
        pd.DataFrame([asdict(trade) for trade in cache[best_config]]).to_csv(
            OUT / "best_discovery_trades.csv", index=False
        )

    columns = [
        "a_mult", "entry_mode", "rr", "extra_cost",
        "median_entry_spread", "p90_entry_spread",
        "y2024_trades", "y2024_win_rate", "y2024_profit_factor", "y2024_avg_r", "y2024_max_dd_r",
        "y2025_trades", "y2025_win_rate", "y2025_profit_factor", "y2025_avg_r", "y2025_max_dd_r",
        "y2026_trades", "y2026_win_rate", "y2026_profit_factor", "y2026_avg_r", "y2026_max_dd_r",
        "all_trades", "all_profit_factor", "all_avg_r", "all_max_dd_r",
    ]
    report = [
        "# XAUUSD ACD True BID/ASK Forward Test",
        "",
        f"Data: dukascopy-node BID and ASK M1, {START_TEST} through {END_TEST}.",
        "Execution: buy-stop triggers and fills on ASK; long stop/target/time exit execute on BID; same-M5 ambiguity is counted as stop.",
        "Fixed filters: Tue-Fri, long only, NY 09:30-09:45 OR, EMA200, no C reversal, OR/ATR 0.05-0.35.",
        "Selection: 2024 only. Validation: 2025. Forward test: 2026 through July 16.",
        f"Merged M1 rows: {len(m1):,}; median price: {median_price:.3f}; median spread: {median_spread:.3f}; spread P90: {p90_spread:.3f}.",
        "",
        "## Top 15 selected only on 2024",
        "",
        discovery.head(15)[columns].to_markdown(index=False, floatfmt=".3f") if not discovery.empty else "No 2024-positive configuration.",
        "",
        "## Positive in both 2025 and 2026 YTD",
        "",
        f"Count: {len(robust)}",
        "",
        robust.head(30)[columns].to_markdown(index=False, floatfmt=".3f") if not robust.empty else "None passed.",
    ]
    (OUT / "bidask_report.md").write_text("\n".join(report), encoding="utf-8")
    (OUT / "data_info.json").write_text(
        json.dumps(
            {
                "test_start": str(START_TEST),
                "test_end": str(END_TEST),
                "m1_rows": int(len(m1)),
                "m5_rows": int(len(bars_utc)),
                "median_price": median_price,
                "median_spread": median_spread,
                "p90_spread": p90_spread,
                "configs": len(configs),
                "forward_positive": int(len(robust)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print((OUT / "bidask_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
