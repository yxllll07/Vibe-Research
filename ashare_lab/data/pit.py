"""按时点读取(point-in-time)。

所有研究代码读数据都经过 AsOfView:只返回 available_at <= 决策时刻 的行;
显式点名要某张表某天的数据而它在决策时刻还不可见,直接抛 LeakageError,不静默返回空。
trade_date 不等于信息可用时间:D 日日线按配置在 D 日 20:00 后才可见,T 日 9:26 的决策看不到 T 日日线。
"""
from __future__ import annotations

from datetime import date, datetime, time as dtime

import pandas as pd

from ..config import TZ


class LeakageError(RuntimeError):
    """请求了决策时刻尚不可见的信息。"""


def d_postclose(d: date, at: dtime = dtime(21, 0)) -> datetime:
    """D 日盘后冻结候选的决策时刻。"""
    return datetime.combine(d, at, TZ)


def auction_decision(t: date) -> datetime:
    """T 日 9:25 最终竞价快照最早可取的时刻(9:26),T 日二阶段过滤的决策时刻。"""
    return datetime.combine(t, dtime(9, 26), TZ)


class AsOfView:
    def __init__(self, tables: dict[str, pd.DataFrame]) -> None:
        for name, df in tables.items():
            if "available_at" not in df.columns:
                raise ValueError(f"表 {name} 没有 available_at,不能进入按时点视图")
        self._tables = tables

    def table(self, name: str, decision_time: datetime) -> pd.DataFrame:
        df = self._tables[name]
        return df[df["available_at"] <= pd.Timestamp(decision_time)]

    def rows_for_date(self, name: str, trade_date: date, decision_time: datetime) -> pd.DataFrame:
        df = self._tables[name]
        rows = df[df["trade_date"] == trade_date]
        late = rows[rows["available_at"] > pd.Timestamp(decision_time)]
        if not late.empty:
            raise LeakageError(
                f"{name} 的 {trade_date} 数据在 {late['available_at'].min()} 才可见,"
                f"决策时刻 {decision_time} 不能使用")
        return rows

    def history(self, name: str, decision_time: datetime, lookback_days: int | None = None) -> pd.DataFrame:
        df = self.table(name, decision_time)
        if lookback_days is not None and not df.empty:
            dates = sorted(df["trade_date"].unique())[-lookback_days:]
            df = df[df["trade_date"].isin(dates)]
        return df
