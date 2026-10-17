"""单日复盘:以信号日 D 运行交易系统 v1,追踪 T 日执行与持有到卖出,再做事后诊断。

事后诊断只用于分析不足,不能反过来当作该日的决策依据;任何据此提出的改动都要在别的日期 / 前向数据上验证。
每笔交易单独模拟(假设空仓起步、每笔 10% 权益),与组合回测的槽位占用不同。
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..build import features_dir
from ..candidates.modes import mode_masks
from ..config import Config
from ..data.calendar import TradingCalendar
from ..data.ingest import calendar_path
from ..evaluation.models import ROUND_TRIP_COST, TARGETS, add_trade_targets
from ..evaluation.proxy_backtest import HOLD_DAYS, STOP_PCT, _bars
from ..rules.execution import Broker
from .daily import MODE_NAMES, STATE_NAMES
from .strategy import Strategy, make_picks

CAPITAL = 1_000_000
POSITION = 0.10


def simulate_trade(code: str, mode: str, bars: dict, cal: TradingCalendar, t: date) -> dict:
    b = Broker(cash=CAPITAL)
    bar = bars.get((t, code))
    if bar is None:
        return {"status": "无 T 日行情(停牌)"}
    fill = b.buy_at_open(code, bar, CAPITAL * POSITION)
    if fill is None:
        return {"status": f"未成交:{b.rejections[-1].reason}", "t_open": bar.open, "t_up_limit": bar.band.up}
    due = cal.shift(t, HOLD_DAYS[mode])
    stop = fill.price * (1 - STOP_PCT[mode]) if mode in STOP_PCT else None
    d = t
    path = []
    for _ in range(30):
        d = cal.next(d)
        bd = bars.get((d, code))
        if bd is None:
            path.append(f"{d} 无行情")
            continue
        out = None
        if stop is not None and d < due:
            out = b.stop_loss(code, bd, stop)
            if out is not None:
                path.append(f"{d} 止损")
        if out is None and d >= due:
            out = b.sell_at_open(code, bd)
            if out is None:
                path.append(f"{d} 卖不出:{b.rejections[-1].reason}")
        if out is not None:
            net = (out.price - fill.price) * out.shares - out.fee - fill.fee
            return {"status": "已平仓", "entry": fill.price, "entry_date": t, "exit": out.price, "exit_date": d,
                    "shares": fill.shares, "net_pnl": round(net, 2), "net_return": net / (fill.price * fill.shares + fill.fee),
                    "stop": stop, "path": path}
    return {"status": "30 个交易日内未能卖出", "entry": fill.price, "path": path}


def _a_base_reason(r: pd.Series) -> str:
    if not r["eligible"]:
        return "不在股票池(ST / 停牌 / 非主板 / 新股)"
    if r["touched_up"]:
        return "D 日已触及涨停(不属 A)"
    if not r["ret_1"] > 0:
        return f"D 日未收涨({r['ret_1']:+.2%})"
    if not r["amount_ratio_5"] > 1.0:
        return f"D 日未放量(成交额 / 5 日均 {r['amount_ratio_5']:.2f})"
    return "在 A 池"


def review_day(cfg: Config, d: date) -> Path:
    strategy = Strategy.load(cfg)
    cal = TradingCalendar.load(calendar_path(cfg.data_root))
    feats_all = pd.read_parquet(features_dir(cfg) / "features.parquet")
    t = cal.next(d)
    window = cal.between(cal.shift(d, -1), cal.shift(d, 12)) if cal.offset(d, cal.days[-1]) >= 12 else cal.between(cal.shift(d, -1), cal.days[-1])
    feats = feats_all[feats_all["trade_date"].isin(window)]
    if feats[feats["trade_date"] == t].empty:
        raise RuntimeError(f"没有 {t} 的行情,无法复盘 {d} 的计划")
    bars = _bars(feats)
    pool, state = make_picks(strategy, feats_all, [d])
    st = int(state[d])
    day = feats[feats["trade_date"] == d].set_index("ts_code")
    tday = feats[feats["trade_date"] == t].set_index("ts_code")
    labels = feats_all[feats_all["trade_date"].isin(window)][["trade_date", "ts_code"]].rename(columns={"trade_date": "signal_date"})
    lab = pd.read_parquet(features_dir(cfg) / "labels.parquet")
    lab = lab[lab["signal_date"] == d]
    lab = add_trade_targets(lab, feats).set_index("ts_code")
    del labels

    masks = mode_masks(day.reset_index())
    from ..evaluation.rule_mining import rule_scores
    sub_all = day.reset_index()
    L = [f"# 单日复盘:信号日 {d} → 执行日 {t}", "",
         f"> 交易系统 v1(策略哈希 `{strategy.hash}`),{'周期过滤开启' if strategy.spec['params'].get('regime_gate') else '周期过滤关闭'}。",
         "> 事后诊断只用于找不足,不是当日可见信息;单笔独立模拟(空仓起步、10% 权益),口径与纸面账本相同。", "",
         "## 1. 市场", ""]
    mk = day.iloc[0]
    L += [f"- 状态 **{st}({STATE_NAMES.get(st, '?')})**",
          f"- 主板非 ST 涨停 {int(mk['mkt_limit_up'])} 家,跌停 {int(mk['mkt_limit_down'])} 家,"
          f"炸板率 {mk['mkt_broke_rate']:.0%},上涨占比 {mk['mkt_up_ratio']:.0%},最高连板 {int(mk['mkt_max_height'])},"
          f"昨日首板晋级率 {mk['mkt_promo_1to2']:.0%}" if pd.notna(mk['mkt_promo_1to2']) else "",
          f"- T 日({t})涨停 {int(tday.iloc[0]['mkt_limit_up'])} 家,上涨占比 {tday.iloc[0]['mkt_up_ratio']:.0%}", ""]

    L += ["## 2. 系统计划与执行", "",
          "| 槽位 | 模式 | 代码 | 名称 | 分数 | 执行 | 买入 | 卖出 | 净收益 |", "|---|---|---|---|---|---|---|---|---|"]
    trades = []
    for r in pool.dropna(subset=["slot"]).itertuples(index=False):
        res = simulate_trade(r.ts_code, r.mode, bars, cal, t)
        trades.append((r, res))
        buy = f"{res['entry']:.2f}" if "entry" in res else "—"
        sell = f"{res['exit_date']} {res['exit']:.2f}" if "exit" in res else "—"
        ret = f"{res['net_return']:+.2%}" if "net_return" in res else "—"
        L.append(f"| {'主动' if r.slot == 'active' else '核心'} | {r.mode} | {r.ts_code} | {day.loc[r.ts_code, 'name']} | "
                 f"{r.m0_score:.3f} | {res['status']} | {buy} | {sell} | {ret} |")
    closed = [x for _, x in trades if "net_return" in x]
    if closed:
        L.append(f"\n合计:{len(closed)} 笔成交,等权平均净收益 {np.mean([x['net_return'] for x in closed]):+.2%};"
                 f"按 10% 仓位折算账户 {sum(x['net_pnl'] for x in closed) / CAPITAL:+.2%}。")
    for r, res in trades:
        if res.get("path"):
            L.append(f"- {r.ts_code} 持有期事件:{';'.join(res['path'])}")
    L.append("")

    L += ["## 3. 各模式池内:我们选的 vs 事后最好", "",
          "目标收益:A/B/D = T 开盘 → T+1 开盘;C/E = T 开盘 → T+4 收盘(未扣费,一字涨停买不到的标 ×)。", ""]
    for m in "ABCDE":
        sub = sub_all[masks[m].fillna(False).values]
        if sub.empty:
            L += [f"### {m} {MODE_NAMES[m]}:池子为空", ""]
            continue
        sc = rule_scores(sub, strategy.rules)[m]
        tab = sub[["ts_code", "name"]].assign(score=sc.values)
        tab["rank"] = tab["score"].rank(ascending=False, method="first").astype(int)
        tgt = TARGETS[m]
        tab["realized"] = tab["ts_code"].map(lab[tgt]) if tgt in lab else np.nan
        tab["unfillable"] = tab["ts_code"].map(lab["t_one_word_up"]).fillna(False).astype(bool)
        tab = tab.sort_values("rank")
        n = len(tab)
        real = tab["realized"]
        top5 = tab.head(5)["realized"].mean()
        best = tab.loc[tab["realized"].where(~tab["unfillable"]).idxmax()] if real.notna().any() else None
        chosen = tab.iloc[0]
        L += [f"### {m} {MODE_NAMES[m]}(池内 {n} 只)", "",
              f"- 池子平均 {real.mean():+.2%}(扣费约 {real.mean() - ROUND_TRIP_COST:+.2%});我们的前 5 平均 {top5:+.2%};"
              f"分数后 5 平均 {tab.tail(5)['realized'].mean():+.2%}",
              f"- 我们的第 1 名 {chosen['ts_code']} {chosen['name']}:{chosen['realized']:+.2%},"
              f"在池内实际收益排第 {int((real > chosen['realized']).sum()) + 1}/{real.notna().sum()}" if pd.notna(chosen["realized"]) else "",
              f"- 事后最好(可买入)的是 {best['ts_code']} {best['name']}:{best['realized']:+.2%},当时分数排第 {int(best['rank'])}"
              if best is not None else "",
              f"- 分数与实际收益的秩相关:{tab['score'].corr(tab['realized'], method='spearman'):+.3f}", ""]
        if best is not None and m in ("A", "B", "E"):
            feats_used = [c["feature"] for c in strategy.rules["modes"][m]["chosen"]]
            comp = sub.set_index("ts_code").loc[[chosen["ts_code"], best["ts_code"]], feats_used].T
            comp.columns = [f"我们选的 {chosen['ts_code']}", f"事后最好 {best['ts_code']}"]
            L += ["| 规则特征 | " + " | ".join(comp.columns) + " |", "|---|---|---|"]
            for f_, row in comp.iterrows():
                L.append(f"| {f_} | " + " | ".join(f"{v:.3f}" for v in row.values) + " |")
            L.append("")

    lu = tday[tday["limit_up_close"].astype(bool) & tday["eligible"].astype(bool)]
    L += ["## 4. T 日涨停股:我们能不能事前抓到", "", f"T 日主板非 ST 收盘涨停 {len(lu)} 只。按 D 日情况分类:", ""]
    rows: list[tuple] = []
    for code in lu.index:
        if code not in day.index:
            rows.append(("D 日无行情", code, "", []))
            continue
        r = day.loc[code]
        in_modes = [m for m in "ABCDE" if bool(masks[m].fillna(False).values[list(day.index).index(code)])]
        rows.append(((",".join(in_modes) + " 池") if in_modes else _a_base_reason(r), code, r["name"], in_modes))
    cat = pd.DataFrame(rows, columns=["D 日归属", "code", "name", "modes"])
    for k, g in cat.groupby("D 日归属"):
        L.append(f"- **{k}**:{len(g)} 只 — " + "、".join(f"{c} {n}" for c, n in zip(g.code.head(8), g.name.head(8)))
                 + (" …" if len(g) > 8 else ""))
    a_hits = cat[cat["modes"].apply(lambda ms: "A" in ms)]
    if len(a_hits):
        sub = sub_all[masks["A"].fillna(False).values]
        sc = pd.Series(rule_scores(sub, strategy.rules)["A"].values, index=sub["ts_code"]).rank(ascending=False)
        L.append(f"- 其中在 A 池里的涨停股,A 规则分数排名:" + "、".join(f"{c} 第 {int(sc[c])}/{len(sc)}" for c in a_hits.code))
    L.append("")
    out = cfg.data_root / "system" / "strategy_v1" / "reviews" / f"review_{d:%Y%m%d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(x for x in L if x is not None) + "\n", encoding="utf-8")
    return out
