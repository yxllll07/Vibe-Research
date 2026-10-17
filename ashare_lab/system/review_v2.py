"""v2 交易者视角单日复盘:情绪周期 → 梯队 → 主线与龙头 → 系统计划 → 执行 → 后续验证 → 与 v1 对比。"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..data.calendar import TradingCalendar
from ..data.ingest import calendar_path
from ..evaluation.proxy_backtest import MAX_EXTEND, _bars
from ..rules.execution import Broker
from ..rules.limits import at_price
from ..structure.theme import SW1_NAMES
from .strategy_v2 import CAPITAL, HOLD_RULES, load_panel, make_picks_v2, structure_dir

MODE_NAMES = {"A": "0→1 首板预判", "B": "1→2 首板接力", "C": "核心龙头首次分歧", "D": "分歧转强", "E": "主线趋势中军"}


def _l2name(code: str, panel_day: pd.DataFrame) -> str:
    l1 = SW1_NAMES.get(str(code)[4:6], "?")
    names = panel_day[(panel_day["sw_l2"] == code) & panel_day["limit_up_close"].astype(bool)]["name"].head(3)
    return f"{l1}·{code[-4:]}({'、'.join(names)})" if len(names) else f"{l1}·{code[-4:]}"


def simulate(code: str, mode: str, bars: dict, cal: TradingCalendar, t: date, rules: dict) -> dict:
    rule = rules[mode]
    b = Broker(cash=CAPITAL)
    bar = bars.get((t, code))
    if bar is None:
        return {"status": "T 日停牌"}
    fill = b.buy_at_open(code, bar, CAPITAL * 0.10)
    if fill is None:
        return {"status": f"未成交:{b.rejections[-1].reason}"}
    due = cal.shift(t, rule["hold"])
    stop = fill.price * (1 - rule["stop"]) if rule.get("stop") else None
    ext, d, notes = 0, t, []
    for _ in range(30):
        d = cal.next(d)
        bd = bars.get((d, code))
        if bd is None:
            continue
        out = None
        if stop is not None and d < due:
            out = b.stop_loss(code, bd, stop)
            if out is not None:
                notes.append(f"{d} 止损")
        if out is None and d >= due:
            yb = bars.get((cal.prev(d), code))
            if rule.get("extend_on_lu") and ext < MAX_EXTEND and yb is not None and yb.band.has_limit \
                    and at_price(yb.close, yb.band.up):
                due, ext = cal.next(d), ext + 1
                notes.append(f"{cal.prev(d)} 收涨停 → 续持")
                continue
            out = b.sell_at_open(code, bd)
        if out is not None:
            net = (out.price - fill.price) * out.shares - out.fee - fill.fee
            return {"status": "已平仓", "entry": fill.price, "exit": out.price, "exit_date": d,
                    "net_return": net / (fill.price * fill.shares + fill.fee), "net_pnl": net, "notes": notes}
    return {"status": "未能卖出"}


def review_day_v2(cfg: Config, d: date) -> Path:
    cal = TradingCalendar.load(calendar_path(cfg.data_root))
    panel = load_panel(cfg)
    sent = pd.read_parquet(structure_dir(cfg) / "sentiment.parquet").set_index("trade_date")
    groups = pd.read_parquet(structure_dir(cfg) / "groups.parquet")
    t = cal.next(d)
    window = cal.between(cal.shift(d, -9), min(cal.shift(d, 8), cal.days[-1]))
    pw = panel[panel["trade_date"].isin(window)]
    bars = _bars(pw)
    pd_ = pw[pw["trade_date"] == d]
    main = pd_[(pd_["board"] == "main") & ~pd_["is_risk_warning"].astype(bool) & ~pd_["is_delisting_period"].astype(bool)]
    L = [f"# 交易者视角复盘(v2):信号日 {d} → 执行日 {t}", "",
         "> 情绪周期 / 主线 / 龙头只用 D 日及以前的数据判定;第 6 节的后续走势是事后验证。单笔独立模拟,10% 仓位。", ""]

    # 1 情绪周期
    L += ["## 1. 情绪周期(近 10 个交易日)", "",
          "| 日期 | 阶段 | 情绪分 | 涨停 | 跌停 | 炸板率 | 昨涨停今均 | 1进2 | 最高板 | 高位大面 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for x in cal.between(cal.shift(d, -9), d):
        r = sent.loc[x]
        L.append(f"| {x:%m-%d} | **{r['stage']}** | {r['score']:.2f} | {int(r['n_limit_up'])} | {int(r['n_limit_down'])} | "
                 f"{r['broke_rate']:.0%} | {r['yday_lu_ret']:+.1%} | {r['promo_1to2']:.0%} | {int(r['max_height'])} {r['max_height_names']} | {int(r['high_loss'])} |")
    stage = sent.loc[d, "stage"]
    L += ["", f"**D 日阶段:{stage}**。", ""]

    # 2 梯队
    L += ["## 2. 连板梯队(D 日 ≥2 板)", ""]
    lad = main[main["limit_streak"] >= 2].sort_values("limit_streak", ascending=False)
    for k, g in lad.groupby("limit_streak", sort=False):
        L.append(f"- {k} 板:" + "、".join(f"{n}({SW1_NAMES.get(str(l)[4:6], '?')})" for n, l in zip(g["name"], g["sw_l2"])))
    L.append("")

    # 3 主线与龙头
    L += ["## 3. 主线与龙头", ""]
    gd = groups[groups["trade_date"] == d].sort_values("ml_rank").head(5)
    L += ["| 排名 | 行业(二级) | 主线 | 3 日加权涨停 | 5 日持续天数 | 最高板 | 3 日涨幅 |", "|---|---|---|---|---|---|---|"]
    for r in gd.itertuples(index=False):
        L.append(f"| {int(r.ml_rank)} | {_l2name(r.sw_l2, main)} | {'✅' if r.is_mainline else ''} | {r.heat:.1f} | "
                 f"{int(r.persist)} | {int(r.max_streak)} | {r.ret3:+.1%} |")
    L.append("")
    for code in gd[gd["is_mainline"]]["sw_l2"]:
        mem = main[(main["sw_l2"] == code) & (main["role"] != "")].sort_values("leader_rank").head(5)
        L.append(f"- **{_l2name(code, main)}**:" + " | ".join(
            f"{r.role} {r.name}(高{int(r.height)}、10日{int(r.lu10)}板、抗分歧{int(r.anti5)})" for r in mem.itertuples()))
    sl = main[main["is_space_leader"].astype(bool)]
    L += [f"- 空间龙:{'、'.join(sl['name'])}", ""]

    # 4 计划
    pool = make_picks_v2(panel, [d])
    from ..candidates.modes_v2 import STAGE_GATE, candidates
    allc = candidates(pd_)
    L += ["## 4. 系统 v2 计划", ""]
    names = pd_.set_index("ts_code")["name"]
    for m in "ABCDE":
        cm = allc[allc["mode"] == m].sort_values("score", ascending=False)
        gate = stage in STAGE_GATE[m]
        top = "、".join(f"{names.get(c)}({s:.2f})" for c, s in zip(cm["ts_code"].head(3), cm["score"].head(3))) or "无"
        L.append(f"- {m} {MODE_NAMES[m]}:{'✅ 开' if gate else f'⛔ {stage}日不做'};候选 {len(cm)} 只,前 3:{top}")
    picks = pool.dropna(subset=["slot"]) if len(pool) else pool
    L.append("")
    if picks.empty:
        L += [f"**NO_TRADE**:{stage}日,没有模式通过情绪开关(或开关通过但无候选)。", ""]

    # 5 执行
    trades = []
    if not picks.empty:
        L += ["## 5. 执行", "", "| 槽位 | 模式 | 股票 | 执行 | 买入 | 卖出 | 净收益 | 过程 |", "|---|---|---|---|---|---|---|---|"]
        for r in picks.itertuples(index=False):
            res = simulate(r.ts_code, r.mode, bars, cal, t, HOLD_RULES)
            trades.append(res)
            L.append(f"| {'主动' if r.slot == 'active' else '核心'} | {r.mode} | {names.get(r.ts_code)} | {res['status']} | "
                     f"{res.get('entry', float('nan')):.2f} | {res.get('exit_date', '')} {res.get('exit', float('nan')):.2f} | "
                     f"{res.get('net_return', float('nan')):+.2%} | {';'.join(res.get('notes', []))} |")
        done = [x for x in trades if "net_return" in x]
        if done:
            L.append(f"\n账户合计(每笔 10% 仓位):{sum(x['net_pnl'] for x in done) / CAPITAL:+.2%}")
        L.append("")

    # 6 后续验证
    # 5b 选票打分:以 D 日收盘为基准,不考虑买卖时点
    fw3 = cal.between(t, min(cal.shift(d, 3), cal.days[-1]))
    px = pw.set_index(["trade_date", "ts_code"])
    dclose = pd_.set_index("ts_code")["close"]

    def score_row(code: str) -> dict:
        def get(day, col):
            try:
                return float(px.loc[(day, code), col])
            except KeyError:
                return float("nan")
        c0 = dclose.get(code, float("nan"))
        highs = [get(x, "high") for x in fw3]
        return {"T收盘": get(t, "close") / c0 - 1, "T最高": get(t, "high") / c0 - 1,
                "3日最高": np.nanmax(highs) / c0 - 1 if highs else float("nan"),
                "T涨停": bool(px.loc[(t, code), "limit_up_close"]) if (t, code) in px.index else False}
    tday_ret = pw[(pw["trade_date"] == t) & pw["eligible"].astype(bool)].set_index("ts_code")["close"] / dclose - 1
    cut = tday_ret.quantile(0.95)
    L += ["## 5b. 选票打分(以 D 日收盘为基准,只看选得准不准)", "",
          f"T 日主板可交易股票:平均收盘 {tday_ret.mean():+.2%},涨幅前 5% 的门槛 {cut:+.2%}。", "",
          "| 模式 | 名次 | 股票 | 入选 | T 收盘 | T 盘中最高 | 3 日内最高 | T 涨停 | 前 5% |", "|---|---|---|---|---|---|---|---|---|"]
    for m in "ABCDE":
        cm = allc[allc["mode"] == m].sort_values("score", ascending=False)
        for rank, code in enumerate(cm["ts_code"].head(3), 1):
            s = score_row(code)
            chosen = "✅" if (not picks.empty and code in set(picks["ts_code"])) else ""
            L.append(f"| {m} | {rank} | {names.get(code)} | {chosen} | {s['T收盘']:+.2%} | {s['T最高']:+.2%} | "
                     f"{s['3日最高']:+.2%} | {'✅' if s['T涨停'] else ''} | {'✅' if s['T收盘'] >= cut else ''} |")
        if len(cm):
            pool = [score_row(c) for c in cm["ts_code"]]
            L.append(f"| {m} | 池平均({len(cm)}) |  |  | {np.nanmean([x['T收盘'] for x in pool]):+.2%} | "
                     f"{np.nanmean([x['T最高'] for x in pool]):+.2%} | {np.nanmean([x['3日最高'] for x in pool]):+.2%} | "
                     f"{np.mean([x['T涨停'] for x in pool]):.0%} |  |")
    L.append("")

    L += ["## 6. 事后验证:主线核心后面怎么走", ""]
    fw = cal.between(d, min(cal.shift(d, 4), cal.days[-1]))
    focus = main[(main["is_mainline"].astype(bool) & main["role"].isin(["龙头", "前排"])) | main["is_space_leader"].astype(bool)]
    path = pw[pw["ts_code"].isin(focus["ts_code"]) & pw["trade_date"].isin(fw)].pivot(index="ts_code", columns="trade_date", values="ret_1")
    st = pw[pw["ts_code"].isin(focus["ts_code"]) & pw["trade_date"].isin(fw)].pivot(index="ts_code", columns="trade_date", values="limit_streak")
    L += ["| 股票 | 角色 | " + " | ".join(f"{x:%m-%d}" for x in fw) + " |", "|---|---|" + "---|" * len(fw)]
    for code in path.index:
        role = "空间龙" if bool(focus.set_index("ts_code").loc[code, "is_space_leader"]) else focus.set_index("ts_code").loc[code, "role"]
        cells = [f"{path.loc[code, x]:+.1%}" + (f"[{int(st.loc[code, x])}板]" if st.loc[code, x] and st.loc[code, x] > 0 else "")
                 if x in path.columns and pd.notna(path.loc[code, x]) else "—" for x in fw]
        L.append(f"| {names.get(code)} | {role} | " + " | ".join(cells) + " |")
    tw = pw[(pw["trade_date"] == t) & (pw["board"] == "main") & ~pw["is_risk_warning"].astype(bool)]
    mkt = tw[~tw["is_suspended"].astype(bool)]["ret_1"].mean()
    ml_codes = set(gd[gd["is_mainline"]]["sw_l2"])
    ml_ret = tw[tw["sw_l2"].isin(ml_codes)]["ret_1"].mean()
    gt = groups[groups["trade_date"] == t].sort_values("lu", ascending=False).head(3)
    L += ["", f"- T 日全市场主板等权 {mkt:+.2%};D 日主线行业等权 {ml_ret:+.2%}",
          f"- T 日涨停最多的行业:" + "、".join(f"{_l2name(c, tw)} {int(n)} 只{'(D 日主线)' if c in ml_codes else ''}"
                                         for c, n in zip(gt["sw_l2"], gt["lu"])), ""]

    out = cfg.data_root / "system" / "strategy_v2" / "reviews" / f"review_{d:%Y%m%d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out
