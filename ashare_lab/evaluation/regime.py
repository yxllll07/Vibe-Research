"""M4 市场状态(周期)与择时:只在训练期拟合状态模型、只在训练期决定哪些状态「可做」。

- 状态模型:D 日市场广度 / 连板梯队 / 晋级率 / 成交额变化(MARKET_FEATURES)标准化后,
  高斯混合 K=4(固定种子)。参数只用训练期日期拟合;每天的状态只由当天特征决定(相当于单向过滤,
  不用平滑后验、不看未来)。不预设「冰点 / 主升」等名字,训练后用各状态的实际结果描述它。
- 可做状态:训练期里该模式「分数前 N 只」扣费后平均收益 > 0 且该状态出现 ≥ 15 天。
- 择时组合:验证期只在可做状态出手,其余日子空仓(收益 0)。
⚠️ 这一步是在看过 eod_v1 盲测结果之后加的(研究方案 M4 原本排在 M3 之后);以验证期为判断依据,
2026-04 ~ 10 的数字只作参考。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from ..features.market import MARKET_FEATURES
from .portfolio_probe import _net
from .models import TARGETS
from .stats import block_bootstrap_ci

SEED = 20261017
K = 4
MIN_STATE_DAYS = 15
TOP_N = 5


def fit_states(market: pd.DataFrame, train_end) -> tuple[pd.Series, pd.DataFrame]:
    m = market.set_index("trade_date")[MARKET_FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)
    tr = m[m.index <= train_end]
    mu, sd = tr.mean(), tr.std().replace(0, 1)
    z = ((m - mu) / sd).fillna(0).clip(-5, 5)
    gmm = GaussianMixture(n_components=K, covariance_type="diag", random_state=SEED, n_init=5)
    gmm.fit(z[z.index <= train_end])
    state = pd.Series(gmm.predict(z), index=z.index, name="state")
    profile = m.assign(state=state)[m.index <= train_end].groupby("state")[
        ["mkt_limit_up", "mkt_broke_rate", "mkt_promo_1to2", "mkt_max_height", "mkt_up_ratio", "mkt_amount_chg_5"]].mean()
    profile["train_days"] = state[state.index <= train_end].value_counts().sort_index()
    return state, profile


def daily_top_n(data: pd.DataFrame, score_col: str, mode: str, n: int = TOP_N) -> pd.Series:
    m = data[(data["mode"] == mode) & data[TARGETS[mode]].notna()].dropna(subset=[score_col]).copy()
    m["net"] = _net(m, mode)
    return (m.sort_values(["trade_date", score_col], ascending=[True, False])
            .groupby("trade_date").head(n).groupby("trade_date")["net"].mean())


def regime_report(data: pd.DataFrame, market: pd.DataFrame, score_col: str, train_end) -> dict:
    state, profile = fit_states(market, train_end)
    out: dict = {"profile": profile, "modes": {}}
    for mode in ("B", "E"):
        res = {}
        series = {sp: daily_top_n(data[data["split"] == sp], score_col, mode) for sp in ("train", "validation", "blind_test")}
        tr = series["train"].to_frame("net").join(state)
        by_state = tr.groupby("state")["net"].agg(["mean", "count"])
        good = by_state[(by_state["mean"] > 0) & (by_state["count"] >= MIN_STATE_DAYS)].index.tolist()
        res["train_by_state"] = by_state
        res["tradeable_states"] = good
        hold = 1 if TARGETS[mode] == "r_open_next_open" else 5
        for sp in ("validation", "blind_test"):
            s = series[sp].to_frame("net").join(state)
            timed = s["net"].where(s["state"].isin(good), 0.0)
            res[sp] = {
                "always": block_bootstrap_ci(s["net"], block=max(5, hold)),
                "timed": block_bootstrap_ci(timed, block=max(5, hold)),
                "days_traded": int(s["state"].isin(good).sum()), "days": int(len(s)),
                "by_state": s.groupby("state")["net"].agg(["mean", "count"]),
                "cum_always": float(np.prod(1 + s["net"].iloc[::hold]) - 1),
                "cum_timed": float(np.prod(1 + timed.iloc[::hold]) - 1),
            }
        out["modes"][mode] = res
    out["state"] = state
    return out
