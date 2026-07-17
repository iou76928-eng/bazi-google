from pathlib import Path

import pandas as pd

import universal_edge_horse_race as core


def read_side(asset: str, side: str) -> pd.DataFrame:
    path = Path("data_universal") / f"{asset}_{side}_m5.csv"
    frame = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing {sorted(missing)}")
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
    ).dropna().sort_index()
    return result[~result.index.duplicated(keep="last")]


def load_asset(asset: str) -> pd.DataFrame:
    bid = read_side(asset, "bid")
    ask = read_side(asset, "ask")
    merged = bid.join(ask, how="inner")
    if len(merged) < 80_000:
        raise RuntimeError(f"{asset}: insufficient merged M5 rows {len(merged)}")
    return merged


core.read_side = read_side
core.load_asset = load_asset

if __name__ == "__main__":
    core.main()
