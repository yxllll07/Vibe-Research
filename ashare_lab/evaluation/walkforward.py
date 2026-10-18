"""逐月前推(walk-forward):每个月的开关表只用该月之前已经收盘的交易学出来,未来数据一律无效。

- 学习:截止日 cutoff = 当月第一个交易日。只用 T 日 < cutoff 的成交结果(T 日收盘已知),
  对每个 (D 日阶段, 模式) 统计「该模式当天排第一、竞价能买到的票」T 日开盘买入的当天收涨率;
  天数 ≥ MIN_DAYS 且收涨率 ≥ 同期全市场收涨率 + MARGIN 才放行,放行的模式按收涨率排优先级。
- 应用:当月每个信号日 D 按学到的表出预选池,T 日竞价到涨停价 / 停牌跳过,取第一只。
- 防泄漏:learn_gates 收到任何 T 日 ≥ cutoff 的行直接抛 LeakageError,不静默丢弃。
MIN_DAYS / MARGIN 在看结果之前定好,不按结果调。
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..candidates import modes_v2, modes_v3
from ..data.pit import LeakageError

MIN_DAYS = 20
MARGIN = 0.03
SUCCESS_MARGIN = 0.0015
MODES = ("A", "B", "C", "D", "E")


def _next_bar(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["ts_code", "trade_date"])
    g = p.groupby("ts_code", sort=False)
    return p.assign(t_date=g["trade_date"].shift(-1), t_open=g["open"].shift(-1), t_close=g["close"].shift(-1),
                    t_up=g["up_limit"].shift(-1))


def outcomes(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (五模式候选 + T 日结果, 每日全市场收涨率)。buyable = T 日有竞价且没开在涨停价。"""
    nb = _next_bar(panel)
    nb["buyable"] = (nb["t_open"] > 0) & ((nb["t_open"] - nb["t_up"]).abs() > 1e-6)
    nb["win"] = nb["t_close"] / nb["t_open"] - 1 > SUCCESS_MARGIN
    nb["t_oc"] = nb["t_close"] / nb["t_open"] - 1
    c = modes_v2.candidates(panel).merge(
        nb[["trade_date", "ts_code", "t_date", "t_open", "t_up", "buyable", "win", "t_oc"]], on=["trade_date", "ts_code"])
    c = c[c["t_date"].notna()]
    m = nb[nb["eligible"].astype(bool) & nb["buyable"] & nb["t_date"].notna()]
    market = m.groupby("trade_date").agg(t_date=("t_date", "first"), mkt_win=("win", "mean")).reset_index()
    return c, market


def learn_gates(hist: pd.DataFrame, market: pd.DataFrame, cutoff: date) -> tuple[dict[str, tuple[str, ...]], pd.DataFrame]:
    if (hist["t_date"] >= cutoff).any() or (market["t_date"] >= cutoff).any():
        raise LeakageError(f"学习开关表时混入了 T 日 ≥ {cutoff} 的结果")
    top = hist[hist["buyable"]].sort_values("score", ascending=False).groupby(["trade_date", "mode"]).head(1)
    cell = top.groupby(["stage", "mode"]).agg(days=("win", "size"), win=("win", "mean")).reset_index()
    base = float(market["mkt_win"].mean())
    cell["allowed"] = (cell["days"] >= MIN_DAYS) & (cell["win"] >= base + MARGIN)
    cell["base"] = base
    gates = {st: tuple(g.sort_values("win", ascending=False)["mode"]) for st, g in cell[cell["allowed"]].groupby("stage")}
    return gates, cell


def preselect_with_gates(day_cands: pd.DataFrame, gates: dict[str, tuple[str, ...]], k: int = 10) -> pd.DataFrame:
    """单个信号日:按学到的表出预选池(放行模式按表里的优先级,同模式按模式分),最多 k 只。"""
    stage = str(day_cands["stage"].iloc[0]) if len(day_cands) else ""
    modes = gates.get(stage, ())
    x = day_cands[day_cands["mode"].isin(modes)].copy()
    x["pri"] = x["mode"].map({m: i for i, m in enumerate(modes)})
    x = x.sort_values(["pri", "score", "ts_code"], ascending=[True, False, True]).drop_duplicates("ts_code")
    x["order"] = np.arange(1, len(x) + 1)
    return x.head(k).drop(columns="pri").reset_index(drop=True)


def apply_gates(day_cands: pd.DataFrame, gates: dict[str, tuple[str, ...]]) -> pd.DataFrame | None:
    """单个信号日:预选池 + 模拟 9:25 取第一只可买的。"""
    x = preselect_with_gates(day_cands, gates)
    if x.empty:
        return None
    auction = {c: float(o) for c, o, b in zip(x["ts_code"], x["t_open"], x["buyable"]) if b}
    pick = modes_v3.auction_pick(x, auction, {}, n=1)
    pick = pick[pick["buy"]]
    return None if pick.empty else pick.iloc[[0]]


def walk_forward(panel: pd.DataFrame, start: date, end: date) -> dict:
    c, market = outcomes(panel)
    months = sorted({(d.year, d.month) for d in c["trade_date"] if start <= d <= end})
    picks, tables = [], []
    for y, mth in months:
        days = sorted(d for d in c["trade_date"].unique() if (d.year, d.month) == (y, mth) and start <= d <= end)
        cutoff = days[0]
        gates, cell = learn_gates(c[c["t_date"] < cutoff], market[market["t_date"] < cutoff], cutoff)
        tables.append(cell.assign(month=f"{y}-{mth:02d}"))
        for d in days:
            r = apply_gates(c[c["trade_date"] == d], gates)
            if r is not None:
                picks.append(r.assign(month=f"{y}-{mth:02d}"))
    pk = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
    mk = market[(market["trade_date"] >= start) & (market["trade_date"] <= end)]
    res = {"range": [str(start), str(end)], "min_days": MIN_DAYS, "margin": MARGIN,
           "signal_days": int(len(mk)), "market_close_up_rate": float(mk["mkt_win"].mean())}
    if not pk.empty:
        res.update({"days_traded": int(len(pk)), "close_up_rate": float(pk["win"].mean()),
                    "avg_open_to_close_net": float(pk["t_oc"].mean() - SUCCESS_MARGIN),
                    "by_stage_mode": {f"{s}-{m}": [int(len(g)), round(float(g["win"].mean()), 3)]
                                      for (s, m), g in pk.groupby(["stage", "mode"])}})
    return {"metrics": res, "picks": pk, "tables": pd.concat(tables, ignore_index=True)}
