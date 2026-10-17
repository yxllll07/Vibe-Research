"""情绪周期:冰点 / 修复 / 发酵 / 高潮 / 分歧 / 退潮 / 震荡。

只用 D 日收盘及以前的数据。统计口径:当日有成交的沪深主板非 ST、非退市整理股票。

日度指标(交易者常看的那几个):
- 涨停家数、跌停家数、炸板率
- 昨日涨停今日平均涨幅(赚钱效应的核心指标)、昨日连板股今日平均涨幅
- 1 进 2 / 2 进 3 / 高位(≥3 板)晋级率、最高板高度及个股
- 高位亏钱效应:昨日 ≥2 板的股票今日跌幅 ≤ −5% 的家数
- 主板成交额

情绪分 = 上述指标在过去 60 个交易日里的分位数(含当日、只看过去)的平均,方向统一为「越高情绪越好」。
阶段判定(按顺序,先命中先用),阈值以分位表达,适应不同时期的成交与活跃度:
1. 高潮:情绪分 ≥ 0.80 且涨停家数分位 ≥ 0.80
2. 分歧:前一日为高潮 / 发酵,且(情绪分较前日下降 ≥ 0.20,或跌停家数分位 ≥ 0.85,或高位亏钱分位 ≥ 0.85)
3. 冰点:情绪分 ≤ 0.20
4. 退潮:情绪分 < 0.45,且(连续两日下降,或前一日为分歧 / 退潮)
5. 修复:前一日为冰点 / 退潮,且情绪分回升
6. 发酵:情绪分 ≥ 0.55
7. 其余为震荡
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 60
MIN_PERIODS = 20
STAGES = ("冰点", "修复", "发酵", "高潮", "分歧", "退潮", "震荡")

# (指标, 方向):方向 +1 表示越大情绪越好
SCORE_PARTS = [
    ("yday_lu_ret", 1), ("n_limit_up", 1), ("promo_1to2", 1), ("broke_rate", -1),
    ("n_limit_down", -1), ("high_loss", -1), ("max_height", 1),
]


def daily_metrics(f: pd.DataFrame) -> pd.DataFrame:
    """f:特征表(含 limit_up_close / limit_streak / prev_streak / ret_1 等)。返回每日一行。"""
    b = f[(f["board"] == "main") & ~f["is_risk_warning"].astype(bool) & ~f["is_delisting_period"].astype(bool)
          & ~f["is_suspended"].astype(bool)]
    rows = []
    for d, x in b.groupby("trade_date", sort=True):
        lu = x["limit_up_close"].astype(bool)
        touched = lu.sum() + x["broke_board"].astype(bool).sum()
        ps = x["prev_streak"]
        y1, y2, y3 = x[ps == 1], x[ps == 2], x[ps >= 3]
        ylu = x[ps >= 1]
        high = x[ps >= 2]
        top = x[x["limit_streak"] == x["limit_streak"].max()] if lu.any() else x.iloc[0:0]
        rows.append({
            "trade_date": d,
            "n_limit_up": int(lu.sum()),
            "n_limit_down": int(x["limit_down_close"].astype(bool).sum()),
            "broke_rate": x["broke_board"].astype(bool).sum() / touched if touched else np.nan,
            "yday_lu_ret": float(ylu["ret_1"].mean()) if len(ylu) else np.nan,
            "yday_lb_ret": float(high["ret_1"].mean()) if len(high) else np.nan,
            "promo_1to2": float(y1["limit_up_close"].mean()) if len(y1) else np.nan,
            "promo_2to3": float(y2["limit_up_close"].mean()) if len(y2) else np.nan,
            "promo_high": float(y3["limit_up_close"].mean()) if len(y3) else np.nan,
            "max_height": int(x["limit_streak"].max()) if lu.any() else 0,
            "max_height_names": "、".join(top["name"].head(3)) if len(top) else "",
            "high_loss": int((high["ret_1"] <= -0.05).sum()),
            "up_ratio": float((x["ret_1"] > 0).mean()),
            "amount_cny": float(x["amount_cny"].sum()),
        })
    return pd.DataFrame(rows)


def _trailing_pct(s: pd.Series) -> pd.Series:
    """当日值在过去 WINDOW 个交易日(含当日)中的分位;不足 MIN_PERIODS 天为 NA。"""
    def pct(w: np.ndarray) -> float:
        w = w[~np.isnan(w)]
        if len(w) < MIN_PERIODS:
            return np.nan
        return float((w <= w[-1]).mean())
    return s.astype(float).rolling(WINDOW, min_periods=MIN_PERIODS).apply(pct, raw=True)


def sentiment_cycle(metrics: pd.DataFrame) -> pd.DataFrame:
    m = metrics.sort_values("trade_date").reset_index(drop=True).copy()
    parts = []
    for col, sign in SCORE_PARTS:
        p = _trailing_pct(m[col] * sign)
        m[f"pct_{col}"] = p
        parts.append(p)
    m["score"] = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    m["pct_n_limit_down_raw"] = _trailing_pct(m["n_limit_down"])
    m["pct_high_loss_raw"] = _trailing_pct(m["high_loss"])
    stages: list[str] = []
    for i, r in m.iterrows():
        s = r["score"]
        prev = stages[-1] if stages else "震荡"
        ps = m.at[i - 1, "score"] if i >= 1 else np.nan
        pps = m.at[i - 2, "score"] if i >= 2 else np.nan
        if pd.isna(s):
            st = "震荡"
        elif s >= 0.80 and r["pct_n_limit_up"] >= 0.80:
            st = "高潮"
        elif prev in ("高潮", "发酵") and ((pd.notna(ps) and ps - s >= 0.20) or r["pct_n_limit_down_raw"] >= 0.85
                                          or r["pct_high_loss_raw"] >= 0.85):
            st = "分歧"
        elif s <= 0.20:
            st = "冰点"
        elif s < 0.45 and ((pd.notna(ps) and pd.notna(pps) and s < ps < pps) or prev in ("分歧", "退潮")):
            st = "退潮"
        elif prev in ("冰点", "退潮") and pd.notna(ps) and s > ps:
            st = "修复"
        elif s >= 0.55:
            st = "发酵"
        else:
            st = "震荡"
        stages.append(st)
    m["stage"] = stages
    return m
