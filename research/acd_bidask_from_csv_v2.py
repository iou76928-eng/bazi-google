from pathlib import Path

import pandas as pd

import acd_bidask_from_csv as base


def read_side(path: Path, side: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")

    index = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    prefix = side.lower()
    # Use NumPy arrays so pandas does not align RangeIndex source Series
    # against the DatetimeIndex and silently replace all values with NaN.
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


base.read_side = read_side

if __name__ == "__main__":
    base.main()
