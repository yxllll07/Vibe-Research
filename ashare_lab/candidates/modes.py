"""五个交易模式的 D 日入池条件与 M0 规则打分。

⚠️ RULE_VERSION = PLACEHOLDER_v0:研究方案要求以使用者的「五层核心选股 R1」可量化部分作为 M0,
但 R1 的具体定义尚未提供。下面是按方案文字描述写的**占位规则**,只用 D 日收盘及以前可观测的列,
用于把管线跑通;拿到 R1 后替换本文件,并升级 RULE_VERSION。

| 模式 | 方案定义 | 占位入池条件 |
|---|---|---|
| A | T 日 0→1 首板先手,D 日非涨停 | D 日未涨停未触板;放量(成交额 / 5 日均 ≥ 1.5);收在日内高位(≥0.7);所属行业 5 日相对强度前 5;近 20 日有过涨停 |
| B | D 日首板,T 日 1→2 | D 日收盘首板(连板数 = 1) |
| C | 核心龙头首次分歧 | 昨日连板 ≥ 3,D 日未封板且未跌停(首次分歧,D 日收盘可确认) |
| D | D 日分歧 / 走弱,T 日弱转强 | D 日炸板;或昨日涨停而 D 日收跌(未跌停) |
| E | 趋势 | 收盘在 20 / 60 日均线之上,20 日涨幅 ≥ 5%,距 20 日高点回撤 ≤ 8%,D 日未涨停 |
"""
from __future__ import annotations

import pandas as pd

RULE_VERSION = "PLACEHOLDER_v0"
MODES = ("A", "B", "C", "D", "E")
ACTIVE_MODES = ("A", "B", "D")
CORE_MODES = ("C", "E")


def _rank(s: pd.Series, by: pd.Series) -> pd.Series:
    return s.groupby(by).rank(pct=True)


def mode_masks(f: pd.DataFrame) -> dict[str, pd.Series]:
    e = f["eligible"].astype(bool)
    not_lu = ~f["limit_up_close"].astype(bool)
    return {
        "A": e & not_lu & ~f["touched_up"].astype(bool) & (f["amount_ratio_5"] >= 1.5)
             & (f["close_pos_in_range"] >= 0.7) & (f["sec_strength_rank"] <= 5) & (f["limit_up_count_20"] >= 1),
        "B": e & (f["limit_streak"] == 1),
        "C": e & (f["prev_streak"] >= 3) & not_lu & ~f["limit_down_close"].astype(bool),
        "D": e & (f["broke_board"].astype(bool)
                  | (f["prev_limit_up"].astype(bool) & (f["ret_1"] < 0) & ~f["limit_down_close"].astype(bool))),
        "E": e & not_lu & (f["close_to_ma_20"] > 0) & (f["close_to_ma_60"] > 0) & (f["ret_20"] >= 0.05)
             & (f["drawdown_20"] >= -0.08),
    }


def m0_scores(f: pd.DataFrame) -> dict[str, pd.Series]:
    """规则分:同一交易日、同一模式内的分位平均(只用 D 日可见列)。"""
    d = f["trade_date"]
    r = lambda c, asc=True: _rank(f[c] if asc else -f[c], d)  # noqa: E731
    return {
        "A": (r("amount_ratio_5") + r("sec_rel_ret_5") + r("close_pos_in_range") + r("ret_5")) / 4,
        "B": (r("sec_limit_up") + r("broke_count_5", asc=False) + r("ret_20", asc=False)) / 3,
        "C": (r("prev_streak") + r("amount_ratio_5")) / 2,
        "D": (r("sec_rel_ret_5") + r("close_pos_in_range")) / 2,
        "E": (r("ret_20") + r("role_ret_5_rank_in_sec")) / 2,
    }


def candidate_rows(f: pd.DataFrame) -> pd.DataFrame:
    """每行 = (trade_date, ts_code, mode, m0_score);同一股票可同时满足多个模式。"""
    masks, scores = mode_masks(f), m0_scores(f)
    parts = []
    for m in MODES:
        sub = f.loc[masks[m].fillna(False), ["trade_date", "ts_code"]].copy()
        sub["mode"] = m
        sub["m0_score"] = scores[m][masks[m].fillna(False)].values
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    out["rule_version"] = RULE_VERSION
    return out


def top_k(cands: pd.DataFrame, score_col: str, k: int = 2) -> pd.DataFrame:
    """每个交易日、每个模式按分数取前 k(主选 + 备选);分数为 NA 的不入选。"""
    c = cands.dropna(subset=[score_col]).sort_values(["trade_date", "mode", score_col, "ts_code"],
                                                       ascending=[True, True, False, True])
    c["rank"] = c.groupby(["trade_date", "mode"]).cumcount() + 1
    return c[c["rank"] <= k].reset_index(drop=True)


def pick_slots(pool: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """主动槽:A/B/D 的主选里取分数最高的一只;核心槽:C/E 的主选各自保留。每模式每天最多新买 1 只。

    不同模式的 M0 分数是各自模式内的分位,不可比;M1 用同一目标(扣费后收益为正的概率)校准后才可比。
    分数并列时按 B > A > D 的固定优先级(写死,不随结果调整)。
    """
    first = pool[pool["rank"] == 1].copy()
    prio = {"B": 0, "A": 1, "D": 2}
    act = first[first["mode"].isin(ACTIVE_MODES)].assign(_p=lambda x: x["mode"].map(prio))
    act = act.sort_values(["trade_date", score_col, "_p"], ascending=[True, False, True]).groupby("trade_date").head(1)
    core = first[first["mode"].isin(CORE_MODES)]
    out = pd.concat([act.drop(columns="_p").assign(slot="active"), core.assign(slot="core")], ignore_index=True)
    return out.sort_values(["trade_date", "slot", "mode"]).reset_index(drop=True)


def no_trade_days(all_days: list, picks: pd.DataFrame) -> list:
    have = set(picks["trade_date"])
    return [d for d in all_days if d not in have]

