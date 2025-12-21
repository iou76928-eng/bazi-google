from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import re

# pip install lunar_python
from lunar_python import Solar


@dataclass
class BaZi:
    year: str
    month: str
    day: str
    hour: str

    def as_tuple(self) -> Tuple[str, str, str, str]:
        return (self.year, self.month, self.day, self.hour)


def parse_datetime(s: str) -> Tuple[int, int, int, int, int]:
    """
    支援：
      - YYYY-MM-DD
      - YYYY-MM-DD HH
      - YYYY-MM-DD HH:MM
      - YYYY/MM/DD
      - YYYY/MM/DD HH:MM
    沒輸入時間 -> 預設 12:00
    """
    s = s.strip().replace("/", "-")
    m = re.fullmatch(
        r"(\d{4})-(\d{1,2})-(\d{1,2})(?:\s+(\d{1,2})(?::(\d{1,2}))?)?",
        s
    )
    if not m:
        raise ValueError("格式錯誤：請用 YYYY-MM-DD 或 YYYY-MM-DD HH:MM（例：1990-01-01 13:30）")

    y = int(m.group(1))
    mo = int(m.group(2))
    d = int(m.group(3))
    hh = int(m.group(4)) if m.group(4) is not None else 12
    mm = int(m.group(5)) if m.group(5) is not None else 0

    if not (1 <= mo <= 12):
        raise ValueError("月份需為 1~12")
    if not (1 <= d <= 31):
        raise ValueError("日期需為 1~31")
    if not (0 <= hh <= 23):
        raise ValueError("小時需為 0~23")
    if not (0 <= mm <= 59):
        raise ValueError("分鐘需為 0~59")

    return y, mo, d, hh, mm


def calc_bazi_8char(y: int, mo: int, d: int, hh: int, mm: int) -> BaZi:
    solar = Solar.fromYmdHms(y, mo, d, hh, mm, 0)
    lunar = solar.getLunar()
    ec = lunar.getEightChar()
    return BaZi(
        year=ec.getYear(),
        month=ec.getMonth(),
        day=ec.getDay(),
        hour=ec.getTime(),
    )


def pretty_print(dt_str: str, bazi: BaZi, used_default_time: bool) -> None:
    print("\n==== 八字排盤 ====")
    print(f"輸入時間：{dt_str}")
    print("")
    print(f"年柱：{bazi.year}")
    print(f"月柱：{bazi.month}")
    print(f"日柱：{bazi.day}")
    print(f"時柱：{bazi.hour}")
    print("")
    print("八字：", " ".join(bazi.as_tuple()))
    print("==================")
    if used_default_time:
        print("提醒：你沒輸入出生時間，時柱是用 12:00 計算，若要準請補上 HH:MM")


def main_loop() -> None:
    print("八字排盤（陽曆/公曆）")
    print("輸入格式：YYYY-MM-DD 或 YYYY-MM-DD HH:MM（例：1990-01-01 13:30）")
    print("輸入 q 離開\n")

    while True:
        s = input("> ").strip()
        if s.lower() in {"q", "quit", "exit"}:
            print("已退出。")
            return

        used_default_time = bool(re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", s))
        try:
            y, mo, d, hh, mm = parse_datetime(s)
            bazi = calc_bazi_8char(y, mo, d, hh, mm)
            pretty_print(s, bazi, used_default_time)
        except Exception as e:
            print(f"\n[錯誤] {e}\n")


if __name__ == "__main__":
    try:
        main_loop()
    finally:
        # ✅ 防止「雙擊執行」時視窗直接關掉（看起來像閃退）
        input("\n按 Enter 鍵離開...")
