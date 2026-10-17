"""五个交易模式:D 日基础池(结构定义)与 M0 规则打分。

基础池只按研究方案的文字定义取「结构上属于这个模式」的股票,不设经过调参的阈值;
池内谁更好,由规则文件(evaluation/rule_mining.py 从训练期数据推导)或 M1 模型打分决定。
全部条件只用 D 日收盘及以前可观测的列。

| 模式 | 方案定义 | 基础池 |
|---|---|---|
| A | T 日 0→1 首板先手,D 日非涨停 | D 日未触及涨停、收涨,且成交额高于 5 日均值(有资金关注的非涨停股) |
| B | D 日首板,T 日 1→2 | D 日收盘首板(连板数 = 1) |
| C | 核心龙头首次分歧 | 昨日连板 ≥ 2,D 日未封板(连板股首次断板,D 日收盘可确认) |
| D | D 日分歧 / 走弱,T 日弱转强 | D 日炸板;或昨日涨停而 D 日收跌(未跌停) |
| E | 趋势 | 收盘在 20 / 60 日均线之上且 20 日收益为正,D 日未涨停 |
"""
from __future__ import annotations

from typing import Any

import pandas as pd

MODES = ("A", "B", "C", "D", "E")
ACTIVE_MODES = ("A", "B", "D")
CORE_MODES = ("C", "E")
PLACEHOLDER_VERSION = "PLACEHOLDER_v0"
RULE_VERSION = PLACEHOLDER_VERSION  # 没有规则文件时的默认值;有规则文件时以文件里的 rule_version 为准


def _b(f: pd.DataFrame, col: str) -> pd.Series:
    return f[col].astype("boolean").fillna(False).astype(bool)


def mode_masks(f: pd.DataFrame) -> dict[str, pd.Series]:
    e = _b(f, "eligible")
    lu, ld = _b(f, "limit_up_close"), _b(f, "limit_down_close")
    return {
        "A": e & ~_b(f, "touched_up") & (f["ret_1"] > 0) & (f["amount_ratio_5"] > 1.0),
        "B": e & (f["limit_streak"] == 1),
        "C": e & (f["prev_streak"] >= 2) & ~lu,
        "D": e & (_b(f, "broke_board") | (_b(f, "prev_limit_up") & (f["ret_1"] < 0) & ~ld)),
        "E": e & ~lu & (f["close_to_ma_20"] > 0) & (f["close_to_ma_60"] > 0) & (f["ret_20"] > 0),
    }


def _placeholder_scores(sub: pd.DataFrame, mode: str) -> pd.Series:
    r = lambda c, asc=True: (sub[c] if asc else -sub[c]).groupby(sub["trade_date"]).rank(pct=True)  # noqa: E731
    return {
        "A": lambda: (r("amount_ratio_5") + r("sec_rel_ret_5") + r("close_pos_in_range") + r("ret_5")) / 4,
        "B": lambda: (r("sec_limit_up") + r("broke_count_5", asc=False) + r("ret_20", asc=False)) / 3,
        "C": lambda: (r("prev_streak") + r("amount_ratio_5")) / 2,
        "D": lambda: (r("sec_rel_ret_5") + r("close_pos_in_range")) / 2,
        "E": lambda: (r("ret_20") + r("role_ret_5_rank_in_sec")) / 2,
    }[mode]()


def candidate_rows(f: pd.DataFrame, rules: dict[str, Any] | None = None) -> pd.DataFrame:
    """每行 = (trade_date, ts_code, mode, m0_score);M0 分数在该模式的池内按日排序。"""
    from ..evaluation.rule_mining import rule_scores

    masks = mode_masks(f)
    parts = []
    for m in MODES:
        sub = f.loc[masks[m].fillna(False)]
        rows = sub[["trade_date", "ts_code"]].copy()
        rows["mode"] = m
        if rules is None:
            rows["m0_score"] = _placeholder_scores(sub, m).values
        else:
            rows["m0_score"] = rule_scores(sub, rules)[m].values
        parts.append(rows)
    out = pd.concat(parts, ignore_index=True)
    out["rule_version"] = rules["rule_version"] if rules else PLACEHOLDER_VERSION
    return out


def top_k(cands: pd.DataFrame, score_col: str, k: int = 2) -> pd.DataFrame:
    """每个交易日、每个模式按分数取前 k(主选 + 备选);分数为 NA 的不入选。"""
    c = cands.dropna(subset=[score_col]).sort_values(["trade_date", "mode", score_col, "ts_code"],
                                                       ascending=[True, True, False, True])
    c["rank"] = c.groupby(["trade_date", "mode"]).cumcount() + 1
    return c[c["rank"] <= k].reset_index(drop=True)


def pick_slots(pool: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """主动槽:A/B/D 的主选里取分数最高的一只;核心槽:C/E 的主选各自保留。每模式每天最多新买 1 只。

    M0 分数是各模式池内的分位,跨模式不可比;M1 输出同一目标(扣费后代理收益为正)的概率,才可跨模式比较。
    分数并列时按 B > A > D 的固定优先级(写死,不随结果调整)。
    """
    first = pool[pool["rank"] == 1].copy()
    prio = {"B": 0, "A": 1, "D": 2}
    act = first[first["mode"].isin(ACTIVE_MODES)].assign(_p=lambda x: x["mode"].map(prio))
    act = act.sort_values(["trade_date", score_col, "_p"], ascending=[True, False, True]).groupby("trade_date").head(1)
    core = first[first["mode"].isin(CORE_MODES)]
    out = pd.concat([act.drop(columns="_p").assign(slot="active"), core.assign(slot="core")], ignore_index=True)
    return out.sort_values(["trade_date", "slot", "mode"]).reset_index(drop=True)
