"""交易系统 v2(TRADER_v2):情绪周期 + 主线 + 龙头 + 按交易逻辑重写的五个模式。

- structure:逐日情绪周期、二级行业主线分、个股龙头分与角色(结果缓存到 <数据根>/structure/)。
- 计划:candidates_v2 → 情绪开关 → 每模式前 2 → 主动槽按阶段优先级取一、核心槽 C/E。
- 退出:A / B / D 次日开盘卖,但仍封涨停就续持(不再涨停的次日开盘卖);C 持有 2 日、−5% 止损;E 持有 5 日、−5% 止损。
⚠️ v2 的规则是在复盘 2026-06-23 / 07-03 之后、按交易者框架写的,规则里没有用任何收益去拟合参数;
评价以验证期为主,2026-04 ~ 10 已揭盲,只作参考。
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ..build import features_dir
from ..candidates import modes_v2
from ..config import Config
from ..data.calendar import TradingCalendar
from ..data.ingest import calendar_path, sectors_path
from ..data.sws import daily_l1
from ..evaluation import proxy_backtest
from ..evaluation.stats import block_bootstrap_ci, daily_returns
from ..structure.sentiment import daily_metrics, sentiment_cycle
from ..structure.theme import add_theme

HOLD_RULES: dict[str, dict] = {
    "A": {"hold": 1, "stop": None, "extend_on_lu": True},
    "B": {"hold": 1, "stop": None, "extend_on_lu": True},
    "D": {"hold": 1, "stop": None, "extend_on_lu": True},
    "C": {"hold": 2, "stop": 0.05},
    "E": {"hold": 5, "stop": 0.05},
}
CAPITAL = 1_000_000


def structure_dir(cfg: Config) -> Path:
    return cfg.data_root / "structure"


def build_structure(cfg: Config) -> Path:
    feats = pd.read_parquet(features_dir(cfg) / "features.parquet")
    members = pd.read_parquet(sectors_path(cfg.data_root))
    feats["sw_l2"] = daily_l1(members, feats, "sw_l2").values
    sent = sentiment_cycle(daily_metrics(feats))
    theme, groups = add_theme(feats)
    out = structure_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    sent.to_parquet(out / "sentiment.parquet", index=False)
    theme.to_parquet(out / "theme.parquet", index=False)
    groups.to_parquet(out / "groups.parquet", index=False)
    return out


def load_panel(cfg: Config) -> pd.DataFrame:
    feats = pd.read_parquet(features_dir(cfg) / "features.parquet")
    sd = structure_dir(cfg)
    theme = pd.read_parquet(sd / "theme.parquet")
    sent = pd.read_parquet(sd / "sentiment.parquet")[["trade_date", "stage", "score"]].rename(columns={"score": "sent_score"})
    p = feats.merge(theme, on=["trade_date", "ts_code"], how="left").merge(sent, on="trade_date", how="left")
    for col in ("is_mainline", "is_space_leader"):  # 不在结构表里的(退市整理等)视为否,不能让 NaN 变成 True
        p[col] = p[col].astype("boolean").fillna(False).astype(bool)
    p["role"] = p["role"].fillna("")
    p = p.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    g = p.groupby("ts_code", sort=False)
    for col in ("role", "leader_score", "is_space_leader", "is_mainline"):
        p[f"prev_{col}"] = g[col].shift(1)
    return p


def make_picks_v2(panel: pd.DataFrame, days: list[date]) -> pd.DataFrame:
    sub = panel[panel["trade_date"].isin(days)]
    return modes_v2.select(modes_v2.candidates(sub))


def history_v2(cfg: Config, start: date, end: date, panel: pd.DataFrame | None = None) -> dict:
    panel = load_panel(cfg) if panel is None else panel
    cal = TradingCalendar.load(calendar_path(cfg.data_root))
    days = [x for x in cal.between(start, end) if x < end]
    pool = make_picks_v2(panel, days)
    picks = pool.dropna(subset=["slot"])
    res = proxy_backtest.run(picks, panel, cal, cal.next(days[0]), end, capital=CAPITAL, hold_rules=HOLD_RULES)
    m = res.metrics
    m["daily_return_bootstrap"] = block_bootstrap_ci(daily_returns(res.equity, CAPITAL))
    stages = panel[panel["trade_date"].isin(days)].groupby("trade_date")["stage"].first()
    m["stage_days"] = stages.value_counts().to_dict()
    m["days_with_any_buy_signal"] = int(picks["trade_date"].nunique())
    return {"metrics": m, "trades": res.trades, "equity": res.equity, "pool": pool}
