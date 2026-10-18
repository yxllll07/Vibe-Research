"""大跌预警探针:能不能在 D 日晚(或 9:25 竞价后)提前认出「T 日开盘买入、当天跌超 2%」的票,并把它们挡在预选池外。

做法(全部逐月前推,只用过去):
- 目标 big = T 日开→收 < −2%;样本 = 竞价能买到的五模式候选。
- 每月用 T 日 < 截止日的候选训练 LightGBM(D 日特征 + 模式;另一版再加竞价跳空),给当月候选打风险分。
- 否决:风险分在同日同模式里排最高的 1/3(敏感性:1/2)不进池,其余照当前规则(walkforward, realized)选票。
结果写在 docs/REALIZABLE.md 第 6 节:整体 AUC 约 0.62,但在真正被放行的格子里只有 0.5 ~ 0.6,否决后可兑现收益反而下降
(+0.73% → +0.50%),没有采用。
"""
from __future__ import annotations

from datetime import date
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..features.market import MARKET_FEATURES
from ..features.sector import SECTOR_FEATURES
from ..features.stock import STOCK_FEATURES
from . import walkforward as wf

STAGES = ["冰点", "修复", "发酵", "高潮", "分歧", "退潮", "震荡"]
THEME = ["ml_score", "ml_rank", "heat", "lu5", "lu10", "anti5", "height", "amt_rank_grp", "leader_score", "role_i",
         "sent_score", "stage_i", "is_mainline"]
PARAMS: dict[str, Any] = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, random_state=7, verbose=-1)


def _frame(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    p = panel.copy()
    p["d_gap"] = p["open"] / p["prev_close"] - 1
    p["d_oc"] = p["close"] / p["open"] - 1
    p["stage_i"] = p["stage"].map({s: i for i, s in enumerate(STAGES)})
    p["role_i"] = p["role"].map({"龙头": 3, "前排": 2, "跟风": 1}).fillna(0)
    feats = list(dict.fromkeys(STOCK_FEATURES + SECTOR_FEATURES + MARKET_FEATURES + THEME + ["d_gap", "d_oc"]))
    c, market = wf.outcomes(panel)
    c = c.merge(p[["trade_date", "ts_code", "close"] + feats], on=["trade_date", "ts_code"], how="left")
    c["gap"] = c["t_open"] / c["close"] - 1
    for m in wf.MODES:
        c[f"m_{m}"] = (c["mode"] == m).astype(float)
    c["big"] = (c["t_oc"] < -0.02).astype(int)
    return c, market, feats + [f"m_{m}" for m in wf.MODES]


def _x(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].astype(float).replace([np.inf, -np.inf], np.nan)


def run(panel: pd.DataFrame, start: date, end: date, use_gap: bool = False, veto: float = 2 / 3) -> dict:
    c, market, cols = _frame(panel)
    cols = cols + (["gap"] if use_gap else [])
    cb = c[c["buyable"]]
    months = sorted({(d.year, d.month) for d in cb["trade_date"] if start <= d <= end})
    scored = []
    for y, mth in months:
        days = sorted(d for d in cb["trade_date"].unique() if (d.year, d.month) == (y, mth) and d <= end)
        tr = cb[cb["t_date"] < days[0]]
        model = lgb.LGBMClassifier(**PARAMS).fit(_x(tr, cols), tr["big"])
        te = c[c["trade_date"].isin(days)]
        scored.append(te.assign(risk=np.asarray(model.predict_proba(_x(te, cols)))[:, 1]))
    s = pd.concat(scored, ignore_index=True)
    s["rq"] = s.groupby(["trade_date", "mode"])["risk"].rank(pct=True)
    sb = s[s["buyable"]]
    picks = []
    for y, mth in months:
        days = sorted(d for d in s["trade_date"].unique() if (d.year, d.month) == (y, mth))
        gates, _ = wf.learn_gates(wf.known_before(c, days[0]), market[market["t_date"] < days[0]], days[0])
        for d in days:
            x = s[(s["trade_date"] == d) & (s["rq"] <= veto)]
            r = wf.apply_gates(x, gates)
            if r is not None:
                picks.append(r)
    pk = pd.concat(picks, ignore_index=True)
    return {"auc_all": float(roc_auc_score(sb["big"], sb["risk"])),
            "auc_by_mode": {m: float(roc_auc_score(g["big"], g["risk"])) for m, g in sb.groupby("mode") if g["big"].nunique() > 1},
            "trades": int(len(pk)), "avg_realized": float(pk["r_close1"].mean()),
            "win_rate": float((pk["r_close1"] > 0).mean()), "big_drop_rate": float((pk["t_oc"] < -0.02).mean())}
