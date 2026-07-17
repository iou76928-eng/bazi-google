from datetime import date

from acd_bidask_forward import candle_url, http_get, decode_day


def main():
    test_day = date(2024, 1, 2)
    for side in ["BID", "ASK"]:
        url = candle_url(test_day, side)
        raw = http_get(url, retries=8, timeout=30)
        frame = decode_day(raw, test_day, side)
        print(side, url, "bytes=", len(raw), "rows=", len(frame))
        if frame.empty:
            raise RuntimeError(f"No decoded {side} data")
        print(frame.head(3).to_string())
        price_col = f"{side.lower()}_close"
        median = float(frame[price_col].median())
        if not 500 <= median <= 10000:
            raise RuntimeError(f"Invalid decoded {side} median price: {median}")
        print(side, "median=", median, "start=", frame.index.min(), "end=", frame.index.max())


if __name__ == "__main__":
    main()
