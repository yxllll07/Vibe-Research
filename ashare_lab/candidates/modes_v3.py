"""v3(EXPECT_v3):看 D 日是为了给 T 日做预期,T 日开盘买入、当天收涨才算选对。

D 日收盘后的情绪阶段决定「T 日大概率是什么环境」,只在预期 T 日升温、而 9:25 竞价还没把溢价吃掉的时候开盘买:

| D 日阶段 | T 日预期(两年样本里的下一日阶段) | T 日开盘买 | 理由 |
|---|---|---|---|
| 冰点 | 九成转修复 | B 首板接力、A 首板预判 | 情绪最冷,竞价高开最少,修复的赚钱效应还没被定价 |
| 修复 | 约一半发酵、四成震荡 | B、A | 升温初期,首板晋级和补涨的预期还在 |
| 震荡 | 近半继续震荡、三成多发酵 | B | 无明显方向,只接主线首板 |
| 退潮 | 过半转修复、近四成继续退潮 | B | 退潮末端,主线首板是最先回暖的地方 |
| 发酵 | 四成多转分歧 | 不买 | 热度已在竞价里兑现,开盘买就是付溢价 |
| 高潮 | 近六成转分歧 | 不买 | 高潮次日分歧,接力等于接在山顶(B 当天收涨率三段都只有 37% ~ 39%) |
| 分歧 | 发酵 / 退潮 / 震荡各约三成 | 不买 | 方向不明,C / D 样本太少、三段结论互相矛盾 |

预选池仍是 TRADER_v2 的五模式候选(只用 D 日数据,D 日晚冻结);T 日 9:25 只抓池内股票的竞价,
竞价已到涨停价(买不到)或停牌的跳过,按顺序取下一只。排序:B 优先于 A,同模式按 v2 模式分。
⚠️ 开关是看过 2024-10 ~ 2026-10 三段数据后定的,没有干净的样本外;只有 2026-10-19 之后的前向记录才算检验。
"""
from __future__ import annotations

import pandas as pd

RULE_VERSION = "EXPECT_v3"
EXPECT: dict[str, tuple[str, tuple[str, ...]]] = {
    "冰点": ("转修复", ("B", "A")),
    "修复": ("发酵或震荡", ("B", "A")),
    "震荡": ("震荡或发酵", ("B",)),
    "退潮": ("修复或继续退潮", ("B",)),
    "发酵": ("易转分歧,不开仓", ()),
    "高潮": ("易转分歧,不开仓", ()),
    "分歧": ("方向不明,不开仓", ()),
}
# v4 = v3 + 分歧日也做 B 首板接力(复盘 6-23、6-26 发现分歧次日主线首板并不差)。
# 只改这一格:新增的 84 个分歧日里排第一的 B 当天收涨率 训练 62% / 验证 50% / 留出 65%,三段都高于同期全市场(46% ~ 48%)。
# 同时试过但没采用:分歧做 E / 分歧做 C、发酵做 A / 发酵做 C、修复日按竞价调整 A / B 顺序 —— 至少一段明显变差或只是噪声。
EXPECT_V4: dict[str, tuple[str, tuple[str, ...]]] = {
    **EXPECT,
    "分歧": ("市场整体弱,主线首板常有回流", ("B",)),
}
RULESETS: dict[str, tuple[str, dict[str, tuple[str, tuple[str, ...]]]]] = {
    "v3": (RULE_VERSION, EXPECT),
    "v4": ("EXPECT_v4", EXPECT_V4),
}
MODE_PRIORITY = {"B": 0, "A": 1}


def allowed_modes(stage: str, rules: str = "v3") -> tuple[str, ...]:
    return RULESETS[rules][1].get(stage, ("未知阶段", ()))[1]


def preselect(c: pd.DataFrame, k: int = 10, rules: str = "v3") -> pd.DataFrame:
    """c:modes_v2.candidates 的输出。返回每个信号日按优先级排好的预选池(每日最多 k 只,一只票只保留优先级最高的模式)。"""
    ok = c[[m in allowed_modes(s, rules) for m, s in zip(c["mode"], c["stage"])]].copy()
    ok["pri"] = ok["mode"].map(MODE_PRIORITY)
    ok = ok.sort_values(["trade_date", "pri", "score", "ts_code"], ascending=[True, True, False, True])
    ok = ok.drop_duplicates(["trade_date", "ts_code"])
    ok["order"] = ok.groupby("trade_date").cumcount() + 1
    ok["rule_version"] = RULESETS[rules][0]
    return ok[ok["order"] <= k].drop(columns="pri").reset_index(drop=True)


def auction_pick(pool: pd.DataFrame, auction: dict[str, float], up_limit: dict[str, float], n: int = 1) -> pd.DataFrame:
    """T 日 9:25 后:pool 为单日预选池(已按 order 排序)。竞价缺失(停牌)或已到涨停价的跳过,取前 n 只可买的。"""
    rows = []
    for r in pool.sort_values("order").itertuples(index=False):
        px = auction.get(r.ts_code)
        up = up_limit.get(r.ts_code)
        if px is None or px <= 0:
            reason = "无竞价(停牌)"
        elif up is not None and px >= up - 1e-6:
            reason = "竞价已到涨停价,买不到"
        else:
            reason = ""
        rows.append({**r._asdict(), "auction_price": px, "up_limit": up, "skip_reason": reason})
    out = pd.DataFrame(rows)
    if out.empty:
        return out.assign(buy=pd.Series(dtype=bool))
    ok = out["skip_reason"] == ""
    out["buy"] = ok & (ok.cumsum() <= n)
    return out
