"""日频代理回测(proxy_daily_only)。⚠️ 只检验信号方向,不代表 9:25 之后真的能成交。

流程(每个交易日 t):
1. 先处理到期卖出(开盘)与核心仓止损;卖不出(跌停 / 停牌 / T+1)记拒单并顺延到下一交易日。
2. 再处理 D=t-1 冻结票池选出的买单(开盘),受槽位上限约束:主动槽 ≤ 50%、核心槽 ≤ 50% 权益;
   单笔名义仓位 = 权益 × position_pct(示例:0.5% 风险预算 / 5% 止损 ≈ 10%)。
3. 收盘按收盘价盯市。
退出(占位,与模式规则一起待 R1 替换):A/B/D 在 T+1 开盘卖;C/E 持有到 T+5 开盘,
期间日线低点触及入场价 -5% 即止损(跳空按开盘价,一字跌停顺延)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from ..data.calendar import TradingCalendar
from ..rules.execution import Bar, Broker
from ..rules.fees import FeeModel
from ..rules.limits import LimitBand

HOLD_DAYS = {"A": 1, "B": 1, "D": 1, "C": 5, "E": 5}
STOP_PCT = {"C": 0.05, "E": 0.05}
LABEL = "proxy_daily_only:日频理想化代理,不代表 9:25 之后可成交"


@dataclass
class Position:
    ts_code: str
    mode: str
    slot: str
    entry_date: date
    entry_price: float
    shares: int
    buy_fee: float
    exit_due: date
    stop: float | None


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    rejections: pd.DataFrame
    metrics: dict = field(default_factory=dict)


def _bars(feats: pd.DataFrame) -> dict[tuple[date, str], Bar]:
    cols = ["trade_date", "ts_code", "open", "high", "low", "close", "prev_close", "volume_shares", "limit_pct",
            "up_limit", "down_limit"]
    out = {}
    for r in feats[cols].itertuples(index=False):
        pct = None if pd.isna(r.limit_pct) else float(r.limit_pct)
        band = LimitBand(pct, None if pct is None else float(r.up_limit), None if pct is None else float(r.down_limit))
        out[(r.trade_date, r.ts_code)] = Bar(r.trade_date, r.open, r.high, r.low, r.close, r.prev_close,
                                             int(r.volume_shares), band)
    return out


def run(picks: pd.DataFrame, feats: pd.DataFrame, cal: TradingCalendar, start: date, end: date, *,
        capital: float = 1_000_000, position_pct: float = 0.10, slot_cap: float = 0.50,
        slippage_bps: float = 10.0, fees: FeeModel | None = None) -> BacktestResult:
    bars = _bars(feats[(feats["trade_date"] >= start)])
    broker = Broker(cash=capital, fees=fees or FeeModel(), slippage_bps=slippage_bps)
    days = [d for d in cal.between(start, end)]
    by_signal = {d: g for d, g in picks.groupby("trade_date")}
    positions: list[Position] = []
    last_close: dict[str, float] = {}
    trades, eq_rows = [], []
    equity_prev = capital

    def exposure(slot: str) -> float:
        return sum(p.shares * last_close.get(p.ts_code, p.entry_price) for p in positions if p.slot == slot)

    for t in days:
        # 1. 卖出与止损
        for p in list(positions):
            bar = bars.get((t, p.ts_code))
            if bar is None:
                broker._reject(t, p.ts_code, "sell", "NO_BAR_SUSPENDED")
                continue
            fill = None
            if p.stop is not None and t > p.entry_date and t < p.exit_due:
                fill = broker.stop_loss(p.ts_code, bar, p.stop)
            if fill is None and t >= p.exit_due:
                fill = broker.sell_at_open(p.ts_code, bar)
            if fill is not None:
                gross = (fill.price - p.entry_price) * fill.shares
                net = gross - fill.fee - p.buy_fee
                trades.append({"ts_code": p.ts_code, "mode": p.mode, "slot": p.slot, "entry_date": p.entry_date,
                               "entry_price": p.entry_price, "exit_date": t, "exit_price": fill.price,
                               "shares": fill.shares, "net_pnl": round(net, 2),
                               "net_return": net / (p.entry_price * p.shares + p.buy_fee),
                               "exit_delayed_days": cal.offset(p.exit_due, t) if t > p.exit_due else 0})
                positions.remove(p)
        # 2. 买入(D = 前一交易日冻结的票池)
        try:
            d = cal.prev(t)
        except (IndexError, KeyError):
            d = None
        for r in (by_signal[d].itertuples(index=False) if d in by_signal else []):
            if any(p.ts_code == r.ts_code for p in positions):
                broker._reject(t, r.ts_code, "buy", "ALREADY_HELD")
                continue
            budget = equity_prev * position_pct
            room = equity_prev * slot_cap - exposure(r.slot)
            if room < budget * 0.5:
                broker._reject(t, r.ts_code, "buy", f"SLOT_{r.slot.upper()}_FULL")
                continue
            bar = bars.get((t, r.ts_code))
            if bar is None:
                broker._reject(t, r.ts_code, "buy", "NO_BAR_SUSPENDED")
                continue
            fill = broker.buy_at_open(r.ts_code, bar, min(budget, room))
            if fill is None:
                continue
            hold = HOLD_DAYS[r.mode]
            try:
                due = cal.shift(t, hold)
            except IndexError:
                due = days[-1]
            stop = fill.price * (1 - STOP_PCT[r.mode]) if r.mode in STOP_PCT else None
            positions.append(Position(r.ts_code, r.mode, r.slot, t, fill.price, fill.shares, fill.fee, due, stop))
        # 3. 盯市
        for p in positions:
            bar = bars.get((t, p.ts_code))
            if bar is not None:
                last_close[p.ts_code] = bar.close
        mv = sum(p.shares * last_close.get(p.ts_code, p.entry_price) for p in positions)
        equity_prev = broker.cash + mv
        eq_rows.append({"trade_date": t, "equity": equity_prev, "cash": broker.cash, "positions": len(positions)})

    equity = pd.DataFrame(eq_rows)
    tr = pd.DataFrame(trades)
    rej = pd.DataFrame([vars(x) for x in broker.rejections])
    return BacktestResult(equity, tr, rej, summarize(equity, tr, rej, picks, start, end, capital))


def summarize(equity: pd.DataFrame, trades: pd.DataFrame, rej: pd.DataFrame, picks: pd.DataFrame,
              start: date, end: date, capital: float) -> dict:
    m: dict = {"label": LABEL, "range": [str(start), str(end)]}
    if equity.empty:
        return m
    e = equity["equity"]
    ret = e.pct_change().fillna(e.iloc[0] / capital - 1)
    dd = e / e.cummax() - 1
    n = len(e)
    m.update({
        "trading_days": n, "final_equity": round(float(e.iloc[-1]), 2),
        "total_return": float(e.iloc[-1] / capital - 1),
        "annualized_return": float((e.iloc[-1] / capital) ** (252 / n) - 1) if n else None,
        "max_drawdown": float(dd.min()),
        "sharpe_daily_annualized": float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else None,
    })
    sig = picks[(picks["trade_date"] >= start) & (picks["trade_date"] <= end)]
    m["signals"] = int(len(sig))
    m["no_trade_signal_days"] = int(n - sig["trade_date"].nunique())
    m["trades"] = int(len(trades))
    if not rej.empty:
        buys = rej[rej["side"] == "buy"]
        m["unfilled_buys"] = int(len(buys))
        m["unfilled_by_reason"] = buys["reason"].value_counts().to_dict()
        m["delayed_sell_events"] = rej[rej["side"] == "sell"]["reason"].value_counts().to_dict()
    if not trades.empty:
        m["win_rate"] = float((trades["net_return"] > 0).mean())
        m["avg_net_return_per_trade"] = float(trades["net_return"].mean())
        wins, losses = trades[trades.net_pnl > 0].net_pnl, trades[trades.net_pnl <= 0].net_pnl
        m["profit_loss_ratio"] = float(wins.mean() / -losses.mean()) if len(wins) and len(losses) and losses.mean() < 0 else None
        m["by_mode"] = {mode: {"trades": int(len(g)), "win_rate": float((g.net_return > 0).mean()),
                               "avg_net_return": float(g.net_return.mean()), "net_pnl": float(g.net_pnl.sum())}
                        for mode, g in trades.groupby("mode")}
    return m
