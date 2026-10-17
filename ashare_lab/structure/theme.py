"""主线与龙头(只用 D 日及以前的数据)。

题材分组:申万二级行业(按当时有效的归属)。没有历史概念板块快照,二级行业是能做到的最细、且不前视的近似
(如「小金属」≈ 钨 / 稀土,「通信设备」≈ 光通信)。

主线分(每个交易日在各二级行业之间比较,0~1):
- 涨停热度:近 3 日涨停数加权(D×1、D−1×0.7、D−2×0.5)  35%
- 持续性:近 5 日里「组内涨停 ≥ 2 只」的天数                20%
- 高度:组内 D 日最高连板                                     20%
- 板块强度:组内等权近 3 日涨幅                               25%
主线 = 当日主线分前 3 且加权涨停热度 ≥ 4。

龙头分(主线内、近 5 日有过涨停的股票之间比较,0~1):
- 高度:D 日连板数(D 日断板则取昨日连板数)              35%
- 辨识度:近 10 日涨停次数                                   25%
- 抗分歧:近 5 日里「组内等权下跌、它却涨停」的天数        20%
- 率先启动:近 10 日内第一次涨停越早越高                   10%
- 容量:D 日成交额在组内的分位                             10%
角色:龙头(第 1)、前排(第 2~3)、跟风(其余有涨停的)。空间龙 = 全市场最高连板。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SW1_NAMES = {
    "11": "农林牧渔", "22": "基础化工", "23": "钢铁", "24": "有色金属", "27": "电子", "28": "汽车", "33": "家用电器",
    "34": "食品饮料", "35": "纺织服饰", "36": "轻工制造", "37": "医药生物", "41": "公用事业", "42": "交通运输",
    "43": "房地产", "45": "商贸零售", "46": "社会服务", "48": "银行", "49": "非银金融", "51": "综合",
    "61": "建筑材料", "62": "建筑装饰", "63": "电力设备", "64": "机械设备", "65": "国防军工", "71": "计算机",
    "72": "传媒", "73": "通信", "74": "煤炭", "75": "石油石化", "76": "环保", "77": "美容护理",
}
TOP_MAINLINES = 3
MIN_HEAT = 4.0


def l1_name(code) -> str:
    return SW1_NAMES.get(str(code)[-2:], "未知") if isinstance(code, str) else "未知"


def _roll_sum(df: pd.DataFrame, col: str, n: int) -> pd.Series:
    return df.groupby("ts_code", sort=False)[col].rolling(n, min_periods=1).sum().reset_index(level=0, drop=True)


def add_theme(f: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """f 需含 sw_l2(按时点)。返回 (逐股逐日的主线 / 龙头字段, 逐日逐二级行业的主线表)。"""
    b = f[(f["board"] == "main") & ~f["is_risk_warning"].astype(bool) & ~f["is_delisting_period"].astype(bool)].copy()
    b = b.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    b["lu"] = b["limit_up_close"].astype(bool).astype(float)
    b["traded"] = ~b["is_suspended"].astype(bool)

    # ---- 组级(二级行业 × 日)
    g = (b[b["traded"] & b["sw_l2"].notna()].groupby(["sw_l2", "trade_date"])
         .agg(n=("ts_code", "size"), lu=("lu", "sum"), ret=("ret_1", "mean"), max_streak=("limit_streak", "max"))
         .reset_index().sort_values(["sw_l2", "trade_date"]))
    gg = g.groupby("sw_l2", sort=False)
    g["heat"] = g["lu"] + 0.7 * gg["lu"].shift(1).fillna(0) + 0.5 * gg["lu"].shift(2).fillna(0)
    g["persist"] = gg["lu"].transform(lambda s: (s >= 2).astype(float).rolling(5, min_periods=1).sum())
    g["ret3"] = gg["ret"].transform(lambda s: np.log1p(s).rolling(3, min_periods=1).sum().pipe(np.expm1))
    day = g.groupby("trade_date")
    g["ml_score"] = (0.35 * day["heat"].rank(pct=True) + 0.20 * day["persist"].rank(pct=True)
                     + 0.20 * day["max_streak"].rank(pct=True) + 0.25 * day["ret3"].rank(pct=True))
    g["ml_rank"] = g.groupby("trade_date")["ml_score"].rank(ascending=False, method="first")
    g["is_mainline"] = (g["ml_rank"] <= TOP_MAINLINES) & (g["heat"] >= MIN_HEAT)
    g["sw_l1_name"] = g["sw_l2"].map(lambda c: SW1_NAMES.get(str(c)[4:6], "未知"))

    # ---- 个股
    b = b.merge(g[["sw_l2", "trade_date", "ret", "ml_score", "ml_rank", "is_mainline", "heat"]].rename(
        columns={"ret": "grp_ret"}), on=["sw_l2", "trade_date"], how="left")
    b = b.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    b["lu5"] = _roll_sum(b, "lu", 5)
    b["lu10"] = _roll_sum(b, "lu", 10)
    b["anti"] = (b["lu"].astype(bool) & (b["grp_ret"] < 0)).astype(float)
    b["anti5"] = _roll_sum(b, "anti", 5)
    # 近 10 日第一次涨停距今的天数(越大 = 启动越早);没有涨停为 NA
    idx = b.groupby("ts_code").cumcount()
    last_lu_idx = idx.where(b["lu"].astype(bool))
    first_in_run = pd.Series(np.nan, index=b.index)
    for code, grp in b.groupby("ts_code", sort=False):
        pos = np.flatnonzero(grp["lu"].to_numpy())
        if len(pos) == 0:
            continue
        arr = np.full(len(grp), np.nan)
        for i in range(len(grp)):
            recent = pos[(pos <= i) & (pos > i - 10)]
            if len(recent):
                arr[i] = i - recent[0]
        first_in_run.loc[grp.index] = arr
    b["days_since_first_lu10"] = first_in_run
    del last_lu_idx
    b["height"] = b["limit_streak"].where(b["limit_streak"] > 0, b["prev_streak"])
    b["amt_rank_grp"] = b.groupby(["trade_date", "sw_l2"])["amount_cny"].rank(pct=True)

    cand = b["lu5"] > 0
    keys = ["trade_date", "sw_l2"]
    r = lambda col: b[cand].groupby(keys)[col].rank(pct=True)  # noqa: E731
    b.loc[cand, "leader_score"] = (0.35 * r("height") + 0.25 * r("lu10") + 0.20 * r("anti5")
                                   + 0.10 * r("days_since_first_lu10") + 0.10 * b.loc[cand, "amt_rank_grp"])
    b["leader_rank"] = b[cand].groupby(keys)["leader_score"].rank(ascending=False, method="first")
    b["role"] = np.select([b["leader_rank"] == 1, b["leader_rank"] <= 3, cand], ["龙头", "前排", "跟风"], default="")
    mx = b[b["traded"]].groupby("trade_date")["limit_streak"].transform("max")
    b["is_space_leader"] = False
    b.loc[b["traded"], "is_space_leader"] = (b.loc[b["traded"], "limit_streak"] == mx) & (mx >= 2)
    cols = ["trade_date", "ts_code", "sw_l2", "grp_ret", "ml_score", "ml_rank", "is_mainline", "heat", "lu5", "lu10",
            "anti5", "days_since_first_lu10", "height", "amt_rank_grp", "leader_score", "leader_rank", "role",
            "is_space_leader"]
    return b[cols], g
