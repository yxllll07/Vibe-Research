"""从训练期数据推导 M0 规则(取代占位规则),只用训练期,验证期只用于挑选、盲测期不看。

做法刻意保持简单、少自由度,降低过拟合:
1. 每个模式的「基础池」由研究方案的结构定义给出(modes.BASE_MASKS),不调阈值。
2. 对一组可解释特征,在训练期逐日计算与目标收益(扣费前代理收益)的截面秩相关(RankIC),
   得到均值、t 值,以及训练期前后两半各自的均值(稳定性:两半同号才保留)。
3. 每个模式取 |t| 最大且稳定的前 K 个特征,方向取 RankIC 的符号;
   M0 分数 = 这些特征当日截面分位(按方向)的等权平均。没有任何拟合系数。
结果写成规则文件(含推导期、数据哈希、每个特征的统计量),版本化提交。
"""
from __future__ import annotations

import json
import math
from datetime import date

import numpy as np
import pandas as pd

from ..candidates.modes import MODES
from .models import TARGETS

CANDIDATE_FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60", "amplitude", "body", "upper_shadow", "lower_shadow",
    "close_pos_in_range", "log_amount", "amount_ratio_5", "amount_ratio_20", "close_to_ma_5", "close_to_ma_20",
    "close_to_ma_60", "drawdown_20", "limit_up_count_20", "broke_count_5", "prev_streak", "limit_streak",
    "sec_rel_ret_1", "sec_rel_ret_5", "sec_limit_up", "sec_limit_up_density", "sec_amount_share_chg_5",
    "sec_strength_rank", "role_ret_5_rank_in_sec", "role_amount_rank_in_sec", "role_rel_ret_5_vs_sec",
    "role_sec_followers",
]
TOP_K = 4
MIN_DAYS = 40
MIN_ABS_T = 2.0


def daily_rank_ic(df: pd.DataFrame, feature: str, target: str, min_n: int = 5) -> pd.Series:
    def ic(g: pd.DataFrame) -> float:
        x, y = g[feature], g[target]
        ok = x.notna() & y.notna() & np.isfinite(x)
        if ok.sum() < min_n or x[ok].nunique() < 2:
            return float("nan")
        return float(x[ok].rank().corr(y[ok].rank()))
    return df.groupby("trade_date").apply(ic, include_groups=False).dropna()


def _tstat(s: pd.Series) -> float:
    return float(s.mean() / s.std() * math.sqrt(len(s))) if len(s) > 2 and s.std() > 0 else float("nan")


def mine_rules(train: pd.DataFrame, train_start: date, train_end: date, data_hash: str) -> dict:
    if train["trade_date"].max() > train_end or train["trade_date"].min() < train_start:
        raise ValueError("规则推导只能用训练期样本")
    mid = train_start + (train_end - train_start) / 2
    spec: dict = {"rule_version": "DATA_R0_v1", "derived_from": [str(train_start), str(train_end)],
                  "data_hash": data_hash, "top_k": TOP_K, "min_abs_t": MIN_ABS_T, "modes": {}}
    for mode in MODES:
        sub = train[train["mode"] == mode]
        target = TARGETS[mode]
        stats = []
        for f in CANDIDATE_FEATURES:
            ics = daily_rank_ic(sub, f, target)
            if len(ics) < MIN_DAYS:
                continue
            first = ics[ics.index <= mid]
            second = ics[ics.index > mid]
            stats.append({"feature": f, "ic_mean": float(ics.mean()), "t": _tstat(ics), "days": int(len(ics)),
                          "ic_first_half": float(first.mean()), "ic_second_half": float(second.mean())})
        st = pd.DataFrame(stats)
        chosen: list[dict] = []
        if not st.empty:
            stable = st[(np.sign(st["ic_first_half"]) == np.sign(st["ic_second_half"])) & (st["t"].abs() >= MIN_ABS_T)]
            for r in stable.reindex(stable["t"].abs().sort_values(ascending=False).index).itertuples(index=False):
                chosen.append({"feature": r.feature, "sign": int(np.sign(r.ic_mean)), "ic_mean": r.ic_mean, "t": r.t,
                               "ic_first_half": r.ic_first_half, "ic_second_half": r.ic_second_half})
                if len(chosen) == TOP_K:
                    break
        spec["modes"][mode] = {
            "target": target, "train_rows": int(len(sub)), "train_days": int(sub["trade_date"].nunique()),
            "chosen": chosen,
            "all_features": st.sort_values("t", key=abs, ascending=False).to_dict("records") if not st.empty else [],
        }
    return spec


def rule_scores(f: pd.DataFrame, spec: dict) -> dict[str, pd.Series]:
    """按规则文件算 M0 分数:选中特征的当日截面分位(按方向)等权平均;某模式没有选中特征则全为 NA。"""
    out = {}
    for mode in MODES:
        chosen = spec["modes"].get(mode, {}).get("chosen", [])
        if not chosen:
            out[mode] = pd.Series(np.nan, index=f.index)
            continue
        parts = []
        for c in chosen:
            x = f[c["feature"]].astype(float).replace([np.inf, -np.inf], np.nan) * c["sign"]
            parts.append(x.groupby(f["trade_date"]).rank(pct=True))
        out[mode] = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    return out


def dump(spec: dict) -> str:
    return json.dumps(spec, ensure_ascii=False, indent=2, default=float)
