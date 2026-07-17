from __future__ import annotations

import concurrent.futures
import itertools
import json
import lzma
import math
import struct
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
SYMBOL = "XAUUSD"
NY = "America/New_York"
START_DOWNLOAD = date(2023, 11, 1)
START_TEST = date(2024, 1, 1)
END_TEST = date(2026, 7, 16)
OUT = Path("artifacts_bidask")
OUT.mkdir(exist_ok=True)
REC_SIZE = 24  # >IIIIIf


@dataclass(frozen=True)
class Config:
    a_mult: float
    entry_mode: str
    rr: float
    extra_cost: float


@dataclass
class Trade:
    date: str
    year: int
    entry_time: str
    exit_time: str
    entry_ask: float
    exit_bid: float
    stop_bid: float
    target_bid: float
    risk: float
    gross_points: float
    net_points: float
    r_net: float
    entry_spread: float
    exit_reason: str


def daterange(start: date, end: date):
    current = start
    while current <= end:
        if current.weekday() < 5:
            yield current
        current += timedelta(days=1)


def candle_url(day: date, side: str) -> str:
    # Dukascopy month directory is zero-indexed.
    return (
        f"{BASE_URL}/{SYMBOL}/{day.year:04d}/{day.month - 1:02d}/"
        f"{day.day:02d}/{side}_candles_min_1.bi5"
    )


def http_get(url: str, retries: int = 5, timeout: int = 30) -> bytes:
    last_error = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "acd-research/1.0"})
            with urlopen(req, timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code == 404:
                return b""
            last_error = exc
        except (URLError, TimeoutError) as exc:
            last_error = exc
        time.sleep(min(16, 2**attempt))
    if last_error is not None:
        raise last_error
    return b""


def infer_factor(raw_close: np.ndarray) -> float:
    finite = raw_close[np.isfinite(raw_close) & (raw_close > 0)]
    if not len(finite):
        return 1000.0
    median_raw = float(np.median(finite))
    candidates = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    valid = []
    for factor in candidates:
        value = median_raw / factor
        if 500 <= value <= 10000:
            valid.append((abs(value - 2500), factor))
    return min(valid)[1] if valid else 1000.0


def decode_day(raw: bytes, day: date, side: str) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    try:
        data = lzma.decompress(raw)
    except lzma.LZMAError:
        return pd.DataFrame()
    count = len(data) // REC_SIZE
    if count <= 0:
        return pd.DataFrame()

    dtype = np.dtype([
        ("seconds", ">u4"),
        ("open", ">u4"),
        ("high", ">u4"),
        ("low", ">u4"),
        ("close", ">u4"),
        ("volume", ">f4"),
    ])
    arr = np.frombuffer(data[: count * REC_SIZE], dtype=dtype)
    factor = infer_factor(arr["close"].astype(float))
    base = pd.Timestamp(datetime(day.year, day.month, day.day, tzinfo=timezone.utc))
    index = base + pd.to_timedelta(arr["seconds"].astype(np.int64), unit="s")
    prefix = side.lower()
    frame = pd.DataFrame(
        {
            f"{prefix}_open": arr["open"].astype(float) / factor,
            f"{prefix}_high": arr["high"].astype(float) / factor,
            f"{prefix}_low": arr["low"].astype(float) / factor,
            f"{prefix}_close": arr["close"].astype(float) / factor,
            f"{prefix}_volume": arr["volume"].astype(float),
        },
        index=pd.DatetimeIndex(index),
    )
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def download_pair(day: date) -> pd.DataFrame:
    bid = decode_day(http_get(candle_url(day, "BID")), day, "BID")
    ask = decode_day(http_get(candle_url(day, "ASK")), day, "ASK")
    if bid.empty or ask.empty:
        return pd.DataFrame()
    merged = bid.join(ask, how="inner")
    merged["source_day"] = str(day)
    return merged


def download_all() -> pd.DataFrame:
    days = list(daterange(START_DOWNLOAD, END_TEST))
    frames: list[pd.DataFrame] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        for i, frame in enumerate(executor.map(download_pair, days), 1):
            if not frame.empty:
                frames.append(frame)
            if i % 100 == 0:
                print(f"downloaded {i}/{len(days)} days, usable={len(frames)}")
    if not frames:
        raise RuntimeError("No Dukascopy BID/ASK candle data downloaded")
    data = pd.concat(frames).sort_index()
    data = data[~data.index.duplicated(keep="last")]
    return data


def resample_5m(m1: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "bid_open": "first", "bid_high": "max", "bid_low": "min", "bid_close": "last", "bid_volume": "sum",
        "ask_open": "first", "ask_high": "max", "ask_low": "min", "ask_close": "last", "ask_volume": "sum",
    }
    bars = m1.resample("5min", label="left", closed="left").agg(aggregations)
    bars = bars.dropna(subset=["bid_open", "bid_high", "bid_low", "bid_close", "ask_open", "ask_high", "ask_low", "ask_close"])
    for field in ["open", "high", "low", "close"]:
        bars[f"mid_{field}"] = (bars[f"bid_{field}"] + bars[f"ask_{field}"]) / 2.0
    bars["spread_open"] = bars["ask_open"] - bars["bid_open"]
    bars["spread_close"] = bars["ask_close"] - bars["bid_close"]
    bars["ema200"] = bars["mid_close"].ewm(span=200, adjust=False).mean()
    return bars


def build_atr(bars_utc: pd.DataFrame) -> pd.Series:
    daily = bars_utc[["mid_high", "mid_low", "mid_close"]].resample("1D").agg(
        {"mid_high": "max", "mid_low": "min", "mid_close": "last"}
    ).dropna()
    prev_close = daily["mid_close"].shift(1)
    tr = pd.concat(
        [
            daily["mid_high"] - daily["mid_low"],
            (daily["mid_high"] - prev_close).abs(),
            (daily["mid_low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()
    atr.index = pd.Index([ts.date() for ts in atr.index])
    return atr.dropna()


def prior_atr(atr: pd.Series, utc_day: date) -> float | None:
    values = atr.loc[atr.index < utc_day]
    return None if values.empty else float(values.iloc[-1])


def session_positions(index: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    sh, sm = map(int, start.split(":"))
    eh, em = map(int, end.split(":"))
    minutes = index.hour * 60 + index.minute
    return np.flatnonzero((minutes >= sh * 60 + sm) & (minutes < eh * 60 + em))


def metrics(trades: list[Trade]) -> dict[str, float]:
    if not trades:
        return {"trades": 0, "win_rate": math.nan, "profit_factor": math.nan, "net_points": 0.0, "avg_r": math.nan, "max_dd_r": math.nan, "max_consecutive_losses": 0}
    pnl = np.array([t.net_points for t in trades], dtype=float)
    rs = np.array([t.r_net for t in trades], dtype=float)
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    equity = np.cumsum(rs)
    peaks = np.maximum.accumulate(np.r_[0.0, equity])
    drawdowns = peaks[1:] - equity
    max_losing = current = 0
    for value in pnl:
        if value < 0:
            current += 1
            max_losing = max(max_losing, current)
        else:
            current = 0
    return {
        "trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean() * 100.0),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else math.inf,
        "net_points": float(pnl.sum()),
        "avg_r": float(rs.mean()),
        "max_dd_r": float(drawdowns.max()) if len(drawdowns) else 0.0,
        "max_consecutive_losses": int(max_losing),
    }


def find_entry(day: pd.DataFrame, positions: np.ndarray, level: float, mode: str):
    if not len(positions):
        return None
    if mode == "touch":
        for pos in positions:
            pos = int(pos)
            if pos <= 0:
                continue
            previous = day.iloc[pos - 1]
            bar = day.iloc[pos]
            # Order is eligible after the previous completed M5 bar.
            if previous.mid_close > previous.ema200 and bar.ask_high >= level:
                entry = max(float(level), float(bar.ask_open))
                return pos, entry
        return None
    if mode == "close1":
        for pos in positions:
            pos = int(pos)
            signal = day.iloc[pos]
            if signal.mid_close > level and signal.mid_close > signal.ema200:
                entry_pos = pos + 1
                if entry_pos >= len(day):
                    return None
                if entry_pos > int(positions[-1]):
                    return None
                return entry_pos, float(day.iloc[entry_pos].ask_open)
        return None
    raise ValueError(mode)


def simulate_trade(day: pd.DataFrame, entry_pos: int, entry_ask: float, stop: float, rr: float, extra_cost: float, day_str: str) -> Trade | None:
    risk = entry_ask - stop
    if not np.isfinite(risk) or risk <= 0:
        return None
    target = entry_ask + risk * rr

    after_entry = day.index[entry_pos:]
    noon_candidates = np.flatnonzero((after_entry.hour * 60 + after_entry.minute) >= 12 * 60)
    if len(noon_candidates):
        time_exit_pos = entry_pos + int(noon_candidates[0])
        time_exit_price = float(day.iloc[time_exit_pos].bid_open)
    else:
        time_exit_pos = len(day) - 1
        time_exit_price = float(day.iloc[time_exit_pos].bid_close)

    exit_price = time_exit_price
    exit_pos = time_exit_pos
    reason = "TIME"

    for pos in range(entry_pos, time_exit_pos + 1):
        bar = day.iloc[pos]
        stop_hit = bar.bid_low <= stop
        target_hit = bar.bid_high >= target
        # Conservative ordering when M5 data cannot determine intrabar sequence.
        if stop_hit:
            exit_price = min(float(stop), float(bar.bid_open))
            exit_pos = pos
            reason = "STOP"
            break
        if target_hit:
            exit_price = float(target)
            exit_pos = pos
            reason = "TP"
            break

    gross = exit_price - entry_ask
    net = gross - extra_cost
    spread = float(day.iloc[entry_pos].ask_open - day.iloc[entry_pos].bid_open)
    return Trade(
        date=day_str,
        year=int(day_str[:4]),
        entry_time=day.index[entry_pos].isoformat(),
        exit_time=day.index[exit_pos].isoformat(),
        entry_ask=float(entry_ask),
        exit_bid=float(exit_price),
        stop_bid=float(stop),
        target_bid=float(target),
        risk=float(risk),
        gross_points=float(gross),
        net_points=float(net),
        r_net=float(net / risk),
        entry_spread=spread,
        exit_reason=reason,
    )


def backtest(bars_ny: pd.DataFrame, atr: pd.Series, config: Config) -> list[Trade]:
    trades: list[Trade] = []
    for local_day, day in bars_ny.groupby(bars_ny.index.date, sort=True):
        if local_day < START_TEST or local_day > END_TEST:
            continue
        if pd.Timestamp(local_day).weekday() not in {1, 2, 3, 4}:  # Tue-Fri
            continue
        day = day.sort_index()
        or_positions = session_positions(day.index, "09:30", "09:45")
        if len(or_positions) < 3:
            continue
        orb = day.iloc[or_positions]
        or_high = float(orb.mid_high.max())
        or_low = float(orb.mid_low.min())
        utc_day = day.index[0].tz_convert("UTC").date()
        daily_atr = prior_atr(atr, utc_day)
        if daily_atr is None or daily_atr <= 0:
            continue
        ratio = (or_high - or_low) / daily_atr
        if not 0.05 <= ratio <= 0.35:
            continue
        level = or_high + daily_atr * config.a_mult
        entry_positions = session_positions(day.index, "09:45", "12:00")
        found = find_entry(day, entry_positions, level, config.entry_mode)
        if found is None:
            continue
        trade = simulate_trade(day, found[0], found[1], or_low, config.rr, config.extra_cost, str(local_day))
        if trade is not None:
            trades.append(trade)
    return trades


def period_metrics(trades: list[Trade], year: int) -> dict[str, float]:
    return metrics([trade for trade in trades if trade.year == year])


def main() -> None:
    m1 = download_all()
    m1.to_pickle(OUT / "xauusd_bidask_m1.pkl")
    bars_utc = resample_5m(m1)
    atr = build_atr(bars_utc)
    bars_ny = bars_utc.tz_convert(NY)

    # Sanity checks before allowing a backtest result.
    median_price = float(bars_utc.mid_close.median())
    median_spread = float(bars_utc.spread_open.median())
    if not 500 <= median_price <= 10000:
        raise RuntimeError(f"Decoded XAUUSD price looks invalid: {median_price}")
    if not 0 < median_spread < 20:
        raise RuntimeError(f"Decoded XAUUSD spread looks invalid: {median_spread}")

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
    cache: dict[Config, list[Trade]] = {}
    for config in configs:
        trades = backtest(bars_ny, atr, config)
        cache[config] = trades
        row = asdict(config)
        row.update({f"all_{key}": value for key, value in metrics(trades).items()})
        for year in [2024, 2025, 2026]:
            row.update({f"y{year}_{key}": value for key, value in period_metrics(trades, year).items()})
        if trades:
            row["median_entry_spread"] = float(np.median([trade.entry_spread for trade in trades]))
            row["p90_entry_spread"] = float(np.quantile([trade.entry_spread for trade in trades], 0.90))
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
        pd.DataFrame([asdict(trade) for trade in cache[best_config]]).to_csv(OUT / "best_discovery_trades.csv", index=False)

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
        f"Data: Dukascopy BID and ASK 1-minute candles, {START_TEST} through {END_TEST}.",
        "Execution: buy-stop triggers on ASK; long stop/target/time exit execute on BID; same-M5 stop/target ambiguity is counted as stop.",
        "Fixed strategy filters: Tue-Fri, long only, NY 09:30-09:45 OR, EMA200, no C reversal, OR/ATR 0.05-0.35.",
        "Selection: 2024 only. Validation: 2025. Forward test: 2026 year-to-date.",
        f"Decoded median XAUUSD price: {median_price:.3f}; median 5-minute opening spread: {median_spread:.3f}.",
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
    (OUT / "data_info.json").write_text(json.dumps({
        "download_start": str(START_DOWNLOAD),
        "test_start": str(START_TEST),
        "test_end": str(END_TEST),
        "m1_rows": int(len(m1)),
        "m5_rows": int(len(bars_utc)),
        "median_price": median_price,
        "median_spread": median_spread,
        "configs": len(configs),
        "forward_positive": int(len(robust)),
    }, indent=2), encoding="utf-8")
    print((OUT / "bidask_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
