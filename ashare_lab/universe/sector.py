"""按时点的板块 / 行业成员。

一条成员记录 = (ts_code, sector_id, provider, effective_from, effective_to, first_known_at)。
查询 D 日成员要同时满足:
- 成员关系在 D 日有效:effective_from <= D < effective_to(effective_to 为空表示仍有效);
- 这条记录在决策时刻已被知道:first_known_at <= decision_time。
后者拦住「用 2026 年才抓到的概念成分回填 2024 年」:没有历史快照证据的概念成员,
first_known_at 就是抓取时刻,早于它的查询一律看不到。
"""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd

COLUMNS = ["ts_code", "sector_id", "provider", "effective_from", "effective_to", "first_known_at"]


def validate_membership(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"板块成员表缺列 {sorted(missing)}")
    if df["first_known_at"].isna().any():
        raise ValueError("板块成员必须有 first_known_at(最早可知时间),缺失就不能用于回测")
    return df


def members_asof(df: pd.DataFrame, d: date, decision_time: datetime) -> pd.DataFrame:
    df = validate_membership(df)
    eff_to = df["effective_to"]
    active = (df["effective_from"] <= d) & (eff_to.isna() | (eff_to > d))
    known = pd.to_datetime(df["first_known_at"]) <= pd.Timestamp(decision_time)
    return df[active & known]
