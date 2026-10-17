"""行业(申万一级)结构特征与个股在行业内的「角色」——只用 D 日及以前可观测的数据。

统计口径与 market.py 一致:当日有成交的沪深主板非风险警示 A 股。行业归属来自按时点查询(sws.daily_l1)。
角色特征只描述 D 日已发生的事(是否行业最高连板、行业内涨幅 / 成交额排名、跟随者数量),
不写任何「将来成为龙头」的信息。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_sector_features(df: pd.DataFrame, l1_col: str = "sw_l1") -> pd.DataFrame:
    out = df.copy()
    base = (out["board"] == "main") & ~out["is_risk_warning"] & ~out["is_delisting_period"] & ~out["is_suspended"]
    b = out[base & out[l1_col].notna()]
    keys = ["trade_date", l1_col]
    g = b.groupby(keys)
    agg = pd.DataFrame({
        "sec_n": g.size(),
        "sec_ret_median": g["ret_1"].median(),
        "sec_up_ratio": g["ret_1"].apply(lambda s: (s > 0).mean()),
        "sec_limit_up": g["limit_up_close"].sum(),
        "sec_broke": g["broke_board"].sum(),
        "sec_max_streak": g["limit_streak"].max(),
        "sec_amount": g["amount_cny"].sum(),
    }).reset_index()
    agg["sec_limit_up_density"] = agg["sec_limit_up"] / agg["sec_n"]
    day_amount = agg.groupby("trade_date")["sec_amount"].transform("sum")
    agg["sec_amount_share"] = agg["sec_amount"] / day_amount
    agg = agg.sort_values(keys)
    gs = agg.groupby(l1_col)
    mkt_med = agg.groupby("trade_date")["sec_ret_median"].transform("median")
    agg["sec_rel_ret_1"] = agg["sec_ret_median"] - mkt_med
    agg["_log"] = np.log1p(agg["sec_ret_median"])
    for n in (3, 5, 10):
        agg[f"sec_ret_{n}"] = gs["_log"].transform(lambda s: s.rolling(n, min_periods=n).sum()).pipe(np.expm1)
        agg[f"sec_rel_ret_{n}"] = agg[f"sec_ret_{n}"] - agg.groupby("trade_date")[f"sec_ret_{n}"].transform("median")
    agg["sec_amount_share_chg_5"] = agg["sec_amount_share"] / gs["sec_amount_share"].transform(
        lambda s: s.rolling(5, min_periods=5).mean()) - 1
    agg["sec_limit_up_5"] = gs["sec_limit_up"].transform(lambda s: s.rolling(5, min_periods=1).sum())
    agg["sec_strength_rank"] = agg.groupby("trade_date")["sec_rel_ret_5"].rank(ascending=False, method="min")
    agg = agg.drop(columns=["_log", "sec_amount"])
    out = out.merge(agg, on=keys, how="left")

    # 个股在行业内的角色(D 日已发生)
    inb = base & out[l1_col].notna()
    gk = out[inb].groupby(keys)
    out.loc[inb, "role_ret_5_rank_in_sec"] = gk["ret_5"].rank(pct=True)
    out.loc[inb, "role_amount_rank_in_sec"] = gk["amount_cny"].rank(pct=True)
    out.loc[inb, "role_is_sec_top_streak"] = (out.loc[inb, "limit_streak"] > 0) & (
        out.loc[inb, "limit_streak"] == out.loc[inb, "sec_max_streak"])
    out.loc[inb, "role_sec_followers"] = np.where(out.loc[inb, "limit_up_close"], out.loc[inb, "sec_limit_up"] - 1, 0)
    out["role_rel_ret_1_vs_sec"] = out["ret_1"] - out["sec_ret_median"]
    out["role_rel_ret_5_vs_sec"] = out["ret_5"] - out["sec_ret_5"]
    out["role_is_sec_top_streak"] = out["role_is_sec_top_streak"].astype("boolean")
    return out


SECTOR_FEATURES = [
    "sec_n", "sec_ret_median", "sec_up_ratio", "sec_limit_up", "sec_broke", "sec_max_streak", "sec_limit_up_density",
    "sec_amount_share", "sec_rel_ret_1", "sec_ret_3", "sec_ret_5", "sec_ret_10", "sec_rel_ret_3", "sec_rel_ret_5",
    "sec_rel_ret_10", "sec_amount_share_chg_5", "sec_limit_up_5", "sec_strength_rank",
    "role_ret_5_rank_in_sec", "role_amount_rank_in_sec", "role_is_sec_top_streak", "role_sec_followers",
    "role_rel_ret_1_vs_sec", "role_rel_ret_5_vs_sec",
]
