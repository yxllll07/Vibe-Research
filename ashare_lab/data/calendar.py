"""交易日历:新浪历史交易日表(akshare tool_trade_date_hist_sina),原始 CSV 落盘后归一化。

交易所提前公布全年休市安排,所以日历在当年年初即可见,不构成前视。
通达信盘后包是否存在会再交叉核对一遍(见 quality.py)。
"""
from __future__ import annotations

import io
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from ..config import TZ
from .raw_store import RawStore

SOURCE = "sina_calendar"
URL = "akshare.tool_trade_date_hist_sina"


def fetch_calendar(store: RawStore) -> pd.DataFrame:
    key = datetime.now(TZ).strftime("%Y%m%d")
    rec = store.get(SOURCE, key)
    if rec is None:
        import akshare as ak  # 只在真正联网时才导入

        df = ak.tool_trade_date_hist_sina()
        data = df.to_csv(index=False).encode()
        rec = store.put(SOURCE, key, ".csv", data, URL, 200)
    return normalize_calendar(pd.read_csv(io.BytesIO(rec.path.read_bytes())))


def normalize_calendar(df: pd.DataFrame) -> pd.DataFrame:
    days = pd.to_datetime(df["trade_date"]).dt.date.drop_duplicates().sort_values().reset_index(drop=True)
    if days.empty:
        raise ValueError("交易日历为空")
    out = pd.DataFrame({"trade_date": days})
    out["next_trade_date"] = out["trade_date"].shift(-1)
    out["prev_trade_date"] = out["trade_date"].shift(1)
    return out


class TradingCalendar:
    """按交易日推进,不用自然日偏移。"""

    def __init__(self, days: list[date]) -> None:
        if not days:
            raise ValueError("交易日历为空")
        self.days = sorted(set(days))
        self._pos = {d: i for i, d in enumerate(self.days)}

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "TradingCalendar":
        return cls(list(df["trade_date"]))

    @classmethod
    def load(cls, path: Path) -> "TradingCalendar":
        return cls.from_frame(pd.read_parquet(path))

    def is_open(self, d: date) -> bool:
        return d in self._pos

    def between(self, start: date, end: date) -> list[date]:
        return [d for d in self.days if start <= d <= end]

    def shift(self, d: date, n: int) -> date:
        """d 必须是交易日;返回往后(n>0)或往前(n<0)第 n 个交易日。"""
        if d not in self._pos:
            raise KeyError(f"{d} 不是交易日")
        i = self._pos[d] + n
        if not 0 <= i < len(self.days):
            raise IndexError(f"{d} 偏移 {n} 个交易日超出日历范围")
        return self.days[i]

    def next(self, d: date) -> date:
        return self.shift(d, 1)

    def prev(self, d: date) -> date:
        return self.shift(d, -1)

    def offset(self, a: date, b: date) -> int:
        """b 相对 a 隔了多少个交易日(两者都须是交易日)。"""
        return self._pos[b] - self._pos[a]
