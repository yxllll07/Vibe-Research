"""逐月前推(walk-forward):每个月的开关表只用该月之前已经收盘的交易学出来,未来数据一律无效。

- 学习:截止日 cutoff = 当月第一个交易日。只用 T 日 < cutoff 的成交结果(T 日收盘已知),
  对每个 (D 日阶段, 模式) 统计「该模式当天排第一、竞价能买到的票」T 日开盘买入的当天收涨率;
  天数 ≥ MIN_DAYS 且收涨率 ≥ 同期全市场收涨率 + MARGIN 才放行,放行的模式按收涨率排优先级。
- 应用:当月每个信号日 D 按学到的表出预选池,T 日竞价到涨停价 / 停牌跳过,取第一只。
- 防泄漏:learn_gates 收到任何 T 日 ≥ cutoff 的行直接抛 LeakageError,不静默丢弃。
MIN_DAYS / MARGIN 在看结果之前定好,不按结果调。

objective="realized"(当前默认):评价改成**能兑现的收益**。A 股 T+1,T 日开盘买入最早 T+1 才能卖,
T 日收涨只是账面。每个候选算两种卖法的净收益(扣 COST):
- open1:T+1 开盘卖;一字跌停(开 = 高 = 跌停价)卖不出,顺延到下一个能卖的开盘;
- close1:T+1 收盘卖;收盘在跌停价或 T+1 停牌则卖不出,顺延到之后能卖的开盘。
卖法固定为 close1。放行条件 = 原来的当天收涨率条件(天数 ≥ MIN_DAYS、收涨率 ≥ 全市场 + MARGIN)**且**该格过去
T+1 收盘卖的平均净收益 ≥ REALIZED_MIN(否决「当天收涨、但隔夜吐回去」的格子);优先级仍按当天收涨率。
只用**卖出日 < 截止日**的结果(卖出那天收盘后才知道),混入卖出日未知或 ≥ 截止日的行直接抛 LeakageError。
试过直接按平均净收益放行(更吵,逐月前推更差),没有采用;对比见 docs/REALIZABLE.md。
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..candidates import modes_v2, modes_v3
from ..data.pit import LeakageError

MIN_DAYS = 20
MARGIN = 0.03
SUCCESS_MARGIN = 0.0015
MODES = ("A", "B", "C", "D", "E")
COST = 0.0015            # 双边佣金 + 印花税 + 过户费,约 0.15%(2023-08-28 印花税减半之后)
COST_BEFORE_20230828 = 0.002  # 之前印花税 0.1%,多 0.05%
STAMP_CUT = date(2023, 8, 28)


def cost_on(sell_date: object) -> float:
    return COST_BEFORE_20230828 if isinstance(sell_date, date) and sell_date < STAMP_CUT else COST
REALIZED_MIN = 0.0       # 否决门槛:过去可兑现平均净收益 < 0 的格子不放行
EXITS = ("open1", "close1")
EXIT = "close1"          # 固定 T+1 收盘卖(每格各自挑卖法试过,更吵、逐月前推更差)
EXIT_NAMES = {"open1": "T+1 开盘卖", "close1": "T+1 收盘卖"}
_AHEAD = 6


def _next_bar(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["ts_code", "trade_date"])
    g = p.groupby("ts_code", sort=False)
    return p.assign(t_date=g["trade_date"].shift(-1), t_open=g["open"].shift(-1), t_close=g["close"].shift(-1),
                    t_up=g["up_limit"].shift(-1))


def _realized(c: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """给候选行补 T+1 两种卖法的净收益与卖出日(看不到卖出的为 NA)。第 k 天 = 信号日之后第 k 个交易行,T = 第 1 天。"""
    p = panel.sort_values(["ts_code", "trade_date"])
    g = p.groupby("ts_code", sort=False)
    ahead = {"trade_date": p["trade_date"]}
    for k in range(2, _AHEAD + 1):
        for col in ("trade_date", "open", "high", "close", "down_limit"):
            ahead[f"{col}_{k}"] = g[col].shift(-k)
    a = pd.DataFrame(ahead).assign(ts_code=p["ts_code"])
    x = c[["trade_date", "ts_code"]].merge(a, on=["trade_date", "ts_code"], how="left")
    eq = lambda u, v: (u - v).abs() < 1e-6  # noqa: E731

    def first_open(start: int) -> tuple[pd.Series, pd.Series]:
        px = pd.Series(np.nan, index=x.index)
        dt = pd.Series(pd.NaT, index=x.index, dtype=object)
        for k in range(_AHEAD, start - 1, -1):  # 倒序覆盖,留下最早能卖的那天
            o, h, dn = x[f"open_{k}"], x[f"high_{k}"], x[f"down_limit_{k}"]
            ok = (o > 0) & ~(eq(o, dn) & eq(h, dn))
            px = px.where(~ok, o)
            dt = dt.where(~ok, x[f"trade_date_{k}"])
        return px, dt

    o_px, o_dt = first_open(2)
    later_px, later_dt = first_open(3)
    c2, dn2 = x["close_2"], x["down_limit_2"]
    sell_close = (c2 > 0) & ~eq(c2, dn2)
    c_px = c2.where(sell_close, later_px)
    c_dt = x["trade_date_2"].where(sell_close, later_dt)
    entry = c["t_open"].to_numpy()
    out = c.copy()
    out["x_open1"] = o_dt.to_numpy()
    out["x_close1"] = c_dt.to_numpy()
    out["r_open1"] = o_px.to_numpy() / entry - 1 - out["x_open1"].map(cost_on).to_numpy()
    out["r_close1"] = c_px.to_numpy() / entry - 1 - out["x_close1"].map(cost_on).to_numpy()
    # 两种卖法都确定之后才算「结果已知」
    both = out["x_open1"].notna() & out["x_close1"].notna()
    out["x_date"] = pd.Series(np.where(both, np.maximum(out["x_open1"].where(both, date.min).astype(object),
                                                        out["x_close1"].where(both, date.min).astype(object)), None),
                              index=out.index)
    return out


def outcomes(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (五模式候选 + T 日结果 + T+1 可兑现收益, 每日全市场收涨率)。buyable = T 日有竞价且没开在涨停价。"""
    nb = _next_bar(panel)
    nb["buyable"] = (nb["t_open"] > 0) & ((nb["t_open"] - nb["t_up"]).abs() > 1e-6)
    nb["win"] = nb["t_close"] / nb["t_open"] - 1 > SUCCESS_MARGIN
    nb["t_oc"] = nb["t_close"] / nb["t_open"] - 1
    c = modes_v2.candidates(panel).merge(
        nb[["trade_date", "ts_code", "t_date", "t_open", "t_up", "buyable", "win", "t_oc"]], on=["trade_date", "ts_code"])
    c = c[c["t_date"].notna()].reset_index(drop=True)
    c = _realized(c, panel)
    m = nb[nb["eligible"].astype(bool) & nb["buyable"] & nb["t_date"].notna()]
    market = m.groupby("trade_date").agg(t_date=("t_date", "first"), mkt_win=("win", "mean")).reset_index()
    return c, market


def learn_gates(hist: pd.DataFrame, market: pd.DataFrame, cutoff: date,
                objective: str = "realized") -> tuple[dict[str, tuple[str, ...]], pd.DataFrame]:
    if (hist["t_date"] >= cutoff).any() or (market["t_date"] >= cutoff).any():
        raise LeakageError(f"学习开关表时混入了 T 日 ≥ {cutoff} 的结果")
    if objective == "realized":
        return _learn_realized(hist, market, cutoff)
    top = hist[hist["buyable"]].sort_values("score", ascending=False).groupby(["trade_date", "mode"]).head(1)
    cell = top.groupby(["stage", "mode"]).agg(days=("win", "size"), win=("win", "mean")).reset_index()
    base = float(market["mkt_win"].mean())
    cell["allowed"] = (cell["days"] >= MIN_DAYS) & (cell["win"] >= base + MARGIN)
    cell["base"] = base
    gates = {st: tuple(g.sort_values("win", ascending=False)["mode"]) for st, g in cell[cell["allowed"]].groupby("stage")}
    cell["exit"] = "close0"
    return gates, cell


def _learn_realized(hist: pd.DataFrame, market: pd.DataFrame,
                    cutoff: date) -> tuple[dict[str, tuple[str, ...]], pd.DataFrame]:
    if hist["x_date"].isna().any() or (hist["x_date"] >= cutoff).any():
        raise LeakageError(f"学习开关表时混入了卖出日未知或 ≥ {cutoff} 的结果")
    top = hist[hist["buyable"]].sort_values("score", ascending=False).groupby(["trade_date", "mode"]).head(1)
    base = float(market["mkt_win"].mean())
    rows = []
    for (st, m), g in top.groupby(["stage", "mode"]):
        r = g[f"r_{EXIT}"]
        rows.append({"stage": st, "mode": m, "days": len(g), "win": float(g["win"].mean()), "exit": EXIT,
                     "realized_mean": float(r.mean()), "realized_win": float((r > 0).mean())})
    cell = pd.DataFrame(rows, columns=["stage", "mode", "days", "win", "exit", "realized_mean", "realized_win"])
    cell["base"] = base
    cell["allowed"] = ((cell["days"] >= MIN_DAYS) & (cell["win"] >= base + MARGIN)
                       & (cell["realized_mean"] >= REALIZED_MIN))
    gates = {st: tuple(g.sort_values("win", ascending=False)["mode"]) for st, g in cell[cell["allowed"]].groupby("stage")}
    return gates, cell


def known_before(c: pd.DataFrame, cutoff: date, objective: str = "realized") -> pd.DataFrame:
    """截止日之前结果已经确定的候选行。"""
    if objective == "realized":
        return c[c["x_date"].notna() & (c["x_date"] < cutoff)]
    return c[c["t_date"] < cutoff]


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


def recent(df: pd.DataFrame, cutoff: date, lookback_days: int | None) -> pd.DataFrame:
    """只保留信号日在截止日之前 lookback_days 个自然日内的行;None = 从数据起点开始的全部历史。"""
    if lookback_days is None:
        return df
    return df[df["trade_date"] >= cutoff - timedelta(days=lookback_days)]


def walk_forward(panel: pd.DataFrame, start: date, end: date, objective: str = "realized",
                 lookback_days: int | None = None, outcomes_cache: tuple[pd.DataFrame, pd.DataFrame] | None = None) -> dict:
    c, market = outcomes_cache if outcomes_cache is not None else outcomes(panel)
    months = sorted({(d.year, d.month) for d in c["trade_date"] if start <= d <= end})
    picks, tables = [], []
    for y, mth in months:
        days = sorted(d for d in c["trade_date"].unique() if (d.year, d.month) == (y, mth) and start <= d <= end)
        cutoff = days[0]
        gates, cell = learn_gates(recent(known_before(c, cutoff, objective), cutoff, lookback_days),
                                  recent(market[market["t_date"] < cutoff], cutoff, lookback_days), cutoff, objective)
        tables.append(cell.assign(month=f"{y}-{mth:02d}"))
        exits = {(s_, m_): e_ for s_, m_, e_ in zip(cell["stage"], cell["mode"], cell["exit"])}
        for d in days:
            r = apply_gates(c[c["trade_date"] == d], gates)
            if r is not None:
                r = r.assign(month=f"{y}-{mth:02d}", exit=exits[(r["stage"].iloc[0], r["mode"].iloc[0])])
                picks.append(r)
    pk = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
    mk = market[(market["trade_date"] >= start) & (market["trade_date"] <= end)]
    res = {"range": [str(start), str(end)], "objective": objective, "min_days": MIN_DAYS, "lookback_days": lookback_days,
           "signal_days": int(len(mk)), "market_close_up_rate": float(mk["mkt_win"].mean())}
    if not pk.empty:
        pk["ret"] = [r[f"r_{e}"] if e in EXITS else r["t_oc"] - COST for r, e in zip(pk.to_dict("records"), pk["exit"])]
        res.update(realized_summary(pk))
        res.update({"days_traded": int(len(pk)), "close_up_rate": float(pk["win"].mean()),
                    "avg_open_to_close_net": float(pk["t_oc"].mean() - SUCCESS_MARGIN),
                    "by_stage_mode": {f"{s}-{m}": [int(len(g)), round(float(g["win"].mean()), 3)]
                                      for (s, m), g in pk.groupby(["stage", "mode"])}})
    return {"metrics": res, "picks": pk, "tables": pd.concat(tables, ignore_index=True)}


def realized_summary(pk: pd.DataFrame) -> dict:
    """可兑现口径:每笔用账户的 slot 比例资金(T+1 收盘卖会和下一笔重叠一天,单笔最多 50%),按交易顺序复利出资金曲线;
    同时给出买入价比开盘价差 0.2% / 0.5% 时的敏感性。"""
    out: dict = {}
    for slip in (0.0, 0.002, 0.005):
        r = (1 + pk["ret"]) / (1 + slip) - 1
        row: dict = {"avg_net_return": float(r.mean()), "win_rate": float((r > 0).mean())}
        for slot in (0.5, 0.2):
            eq = (1 + slot * r.fillna(0)).cumprod()
            row[f"slot{int(slot * 100)}_total_return"] = float(eq.iloc[-1] - 1)
            row[f"slot{int(slot * 100)}_max_drawdown"] = float((eq / eq.cummax() - 1).min())
        out[f"slip_{slip:.3f}"] = row
    out["exit_mix"] = pk["exit"].value_counts().to_dict()
    out["by_stage_mode_realized"] = {f"{s}-{m}": [int(len(g)), round(float(g["ret"].mean()), 4), round(float((g["ret"] > 0).mean()), 3)]
                                     for (s, m), g in pk.groupby(["stage", "mode"])}
    return out
