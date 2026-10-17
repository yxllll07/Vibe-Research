"""历史股票池:由逐日全市场快照推导,天然包含后来退市的股票(退市前每天都在快照里)。

security_master 从日线面板推出:
- first_seen / last_seen:窗口内首末出现日;
- listed_in_window:首次出现那天简称带上市首日标记(N/C)→ 上市日 = first_seen;
  窗口首日就已存在且无标记 → 窗口前已上市(上市日未知但早于窗口);
  窗口中途首次出现却没有标记 → UNVERIFIED_LISTING。实测这类多是窗口开始时正在停牌、之后复牌的老股票
  (通达信包不收录停牌股当日行情);它们按老股处理。若其实是没带标记的新股,上市前 5 日无涨跌幅限制,
  价格会越出计算的涨跌停带,quality 报告的「价格带越界」会把它抓出来。
- delisted_by_window_end:last_seen 早于窗口末日且最后几天处于退市整理期 → 视为已退市。
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from ..rules.limits import REGISTRATION_IPO_NO_LIMIT_DAYS


def build_security_master(panel: pd.DataFrame, window_start: date, window_end: date) -> pd.DataFrame:
    a = panel[panel["security_type"] == "A"].sort_values(["ts_code", "trade_date"])
    g = a.groupby("ts_code")
    first = g.first()
    last = g.last()
    master = pd.DataFrame({
        "board": first["board"],
        "exchange": first["exchange"],
        "first_seen": first["trade_date"],
        "last_seen": last["trade_date"],
        "first_name": first["name"],
        "last_name": last["name"],
        "first_day_marker": first["listing_day_marker"],
        "last_in_delisting_period": last["is_delisting_period"],
    })
    master["listing_status"] = "pre_window"
    mid = master["first_seen"] > window_start
    master.loc[mid & master["first_day_marker"], "listing_status"] = "listed_in_window"
    master.loc[mid & ~master["first_day_marker"], "listing_status"] = "UNVERIFIED_LISTING"
    master["list_date"] = master["first_seen"].where(master["listing_status"] == "listed_in_window")
    master["delisted_by_window_end"] = (master["last_seen"] < window_end) & master["last_in_delisting_period"]
    return master.reset_index()


def listed_trading_days(panel: pd.DataFrame, master: pd.DataFrame) -> pd.Series:
    """每行是该股上市后的第几个交易日(首日 = 1);上市日未知(窗口前上市)返回 NA。"""
    m = master.set_index("ts_code")["listing_status"]
    rank = panel.groupby("ts_code")["trade_date"].rank(method="first").astype("Int64")
    known = panel["ts_code"].map(m) == "listed_in_window"
    return rank.where(known)


def eligible_mask(day: pd.DataFrame, listed_days: pd.Series | None = None) -> pd.Series:
    """D 日可进入研究样本:沪深主板 A 股、非风险警示、非退市整理、未停牌、非上市前 5 日。"""
    mask = (
        (day["security_type"] == "A")
        & (day["board"] == "main")
        & day["exchange"].isin(["SSE", "SZSE"])
        & ~day["is_risk_warning"]
        & ~day["is_delisting_period"]
        & ~day["is_suspended"]
    )
    if listed_days is not None:
        young = listed_days.fillna(10_000) <= REGISTRATION_IPO_NO_LIMIT_DAYS
        mask &= ~young.astype(bool)
    return mask


def forbidden_reason(row: pd.Series) -> str | None:
    if row["security_type"] != "A" or row["board"] != "main":
        return "NOT_MAIN_BOARD_A"
    if row["is_risk_warning"]:
        return "RISK_WARNING"
    if row["is_delisting_period"]:
        return "DELISTING_PERIOD"
    if row["is_suspended"]:
        return "SUSPENDED"
    return None
