"""五个模式按短线交易逻辑重写(v2):情绪周期定开关,主线定方向,龙头定选谁。全部只用 D 日收盘及以前的数据。

| 模式 | 交易逻辑 | 入池条件 | 排序 | 允许的情绪阶段 |
|---|---|---|---|---|
| A 0→1 首板预判 | 主线发酵时,找主线里还没板、正在走强的低位股 | 属主线;D 日未触板;近 5 日无涨停;收涨且收在日内上部(≥0.6);放量(量比5 ≥1.2);20 日涨幅 <30% | 主线分 40% + 组内成交额分位 30% + D 日涨幅分位 30% | 修复、发酵 |
| B 1→2 首板接力 | 只接主线里的首板,前排、换手板优先 | 属主线;D 日首板 | 主线分 45% + 组内成交额分位 35% + 非一字板 20% | 修复、发酵、高潮 |
| C 核心龙头首次分歧 | 主线龙头 / 空间龙第一次断板时低吸 | 昨日为主线龙头(昨日连板 ≥2)或昨日空间龙(≥3 板);D 日未封板、未跌停、跌幅 >−7% | 昨日龙头分 | 发酵、高潮、分歧 |
| D 分歧转强 | 主线核心在分歧日被错杀,博次日转强 | 属主线且为龙头 / 前排;D 日炸板,或昨日涨停而 D 日收跌(未跌停) | 龙头分 50% + 主线分 50% | 分歧、修复、发酵 |
| E 主线趋势中军 | 主线里容量最大、趋势健康的中军 | 主线分排名前 5 的行业;站上 20 / 60 日线;20 日涨幅 5%~50%;组内成交额分位 ≥0.8;D 日未涨停 | 主线分 40% + 组内成交额分位 40% + 20 日涨幅分位 20% | 除退潮、冰点外 |

退潮、冰点:全部不开新仓。主动槽在 A / B / D 里按情绪阶段的优先级取一只(分歧日 D 优先,其余 B > A > D);核心槽取 C、E 各一只。
"""
from __future__ import annotations

import pandas as pd

MODES = ("A", "B", "C", "D", "E")
STAGE_GATE = {
    "A": {"修复", "发酵"},
    "B": {"修复", "发酵", "高潮"},
    "C": {"发酵", "高潮", "分歧"},
    "D": {"分歧", "修复", "发酵"},
    "E": {"修复", "发酵", "高潮", "分歧", "震荡"},
}
ACTIVE_PRIORITY = {"分歧": ["D", "B", "A"]}
DEFAULT_PRIORITY = ["B", "A", "D"]
RULE_VERSION = "TRADER_v2"


def _b(f: pd.DataFrame, col: str) -> pd.Series:
    return f[col].astype("boolean").fillna(False).astype(bool)


def _pct(s: pd.Series, by: pd.Series) -> pd.Series:
    return s.groupby(by).rank(pct=True)


def candidates(f: pd.DataFrame) -> pd.DataFrame:
    """f:特征表 + theme 字段 + stage + 昨日的 role / leader_score / is_space_leader(prev_*)。"""
    e = _b(f, "eligible")
    ml = _b(f, "is_mainline")
    lu, ld = _b(f, "limit_up_close"), _b(f, "limit_down_close")
    d = f["trade_date"]
    out = []

    a = e & ml & ~_b(f, "touched_up") & (f["lu5"] == 0) & (f["ret_1"] > 0) & (f["close_pos_in_range"] >= 0.6) \
        & (f["amount_ratio_5"] >= 1.2) & (f["ret_20"] < 0.30)
    sa = f[a]
    out.append(sa.assign(mode="A", score=0.4 * sa["ml_score"] + 0.3 * sa["amt_rank_grp"]
                         + 0.3 * _pct(sa["ret_1"], sa["trade_date"])))

    bm = e & ml & (f["limit_streak"] == 1)
    sb = f[bm]
    out.append(sb.assign(mode="B", score=0.45 * sb["ml_score"] + 0.35 * sb["amt_rank_grp"]
                         + 0.2 * (~_b(sb, "one_word_up")).astype(float)))

    was_leader = ((f["prev_role"] == "龙头") & _b(f, "prev_is_mainline") & (f["prev_streak"] >= 2)) \
        | (_b(f, "prev_is_space_leader") & (f["prev_streak"] >= 3))
    cm = e & was_leader & ~lu & ~ld & (f["ret_1"] > -0.07)
    sc = f[cm]
    out.append(sc.assign(mode="C", score=sc["prev_leader_score"].fillna(0)))

    dm = e & ml & f["role"].isin(["龙头", "前排"]) & (_b(f, "broke_board") | (_b(f, "prev_limit_up") & (f["ret_1"] < 0) & ~ld))
    sd = f[dm]
    out.append(sd.assign(mode="D", score=0.5 * sd["leader_score"].fillna(0) + 0.5 * sd["ml_score"]))

    em = e & (f["ml_rank"] <= 5) & (f["close_to_ma_20"] > 0) & (f["close_to_ma_60"] > 0) & (f["ret_20"] >= 0.05) \
        & (f["ret_20"] <= 0.50) & (f["amt_rank_grp"] >= 0.8) & ~lu
    se = f[em]
    out.append(se.assign(mode="E", score=0.4 * se["ml_score"] + 0.4 * se["amt_rank_grp"]
                         + 0.2 * _pct(se["ret_20"], se["trade_date"])))

    c = pd.concat(out, ignore_index=True)
    c["gate_ok"] = [st in STAGE_GATE[m] for m, st in zip(c["mode"], c["stage"])]
    c["rule_version"] = RULE_VERSION
    del d
    return c[["trade_date", "ts_code", "mode", "score", "stage", "gate_ok", "rule_version"]]


def select(c: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """通过情绪开关的候选:每模式前 k;主动槽按阶段优先级取一只,核心槽 C、E 各一只。"""
    ok = c[c["gate_ok"]].sort_values(["trade_date", "mode", "score", "ts_code"], ascending=[True, True, False, True])
    ok = ok.assign(rank=ok.groupby(["trade_date", "mode"]).cumcount() + 1)
    pool = ok[ok["rank"] <= k].copy()
    pool["slot"] = None
    for dte, g in pool[pool["rank"] == 1].groupby("trade_date"):
        stage = g["stage"].iloc[0]
        for m in ACTIVE_PRIORITY.get(stage, DEFAULT_PRIORITY):
            hit = g[g["mode"] == m]
            if len(hit):
                pool.loc[hit.index, "slot"] = "active"
                break
        pool.loc[g[g["mode"].isin(["C", "E"])].index, "slot"] = "core"
    return pool.reset_index(drop=True)
