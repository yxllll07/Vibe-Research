"""交易系统 v3(EXPECT_v3):D 日晚冻结预选池并给出 T 日预期,T 日 9:25 只看池内竞价,开盘买入,当天收涨算选对。

- plan-v3 --date D:D 日阶段 → T 日预期 → 允许的模式 → 预选池(最多 10 只,按优先级排好),写 plan.vN.json(只增不改)+ plan.md。
- auction-v3 --date T:只抓预选池内股票的 9:25 竞价(腾讯行情,或 --file 给的 9:25 记录),到涨停价 / 停牌的跳过,取第一只可买的。
- history-v3:按同样规则逐日重放,统计 T 日开盘买入后当天的收涨率(开盘价近似 9:25 竞价价)。
⚠️ 日线里看到竞价后按开盘价成交是理想化:实盘看到 9:25 结果后只能在 9:30 之后按连续竞价价格买入。
"""
from __future__ import annotations

import json
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..candidates import modes_v2, modes_v3
from ..config import TZ, Config
from ..data.auction import validate_auction_record
from ..data.calendar import TradingCalendar
from ..data.ingest import calendar_path
from ..rules.limits import limit_band
from .strategy_v2 import load_panel

MODE_NAMES = {"A": "首板预判", "B": "首板接力", "C": "龙头首次分歧", "D": "分歧转强", "E": "主线中军"}
SUCCESS_MARGIN = 0.0015  # 收盘高于开盘 0.15% 以上(约等于双边费用)才算收涨
QUOTE_URL = "https://qt.gtimg.cn/q="
_WF_GATES: dict[date, dict[str, tuple[str, ...]]] = {}


DEFAULT_RULES = "wf"  # wf = 只用过去已兑现的结果现学开关表,T+1 收盘卖(evaluation/walkforward.py,当前默认)
# wf1 = 上一版:只看当天收涨率学开关表;v3 / v4 = 手写表(看过全样本)
WF_OBJECTIVE = {"wf": "realized", "wf1": "close_up"}


def plans_root(cfg: Config, rules: str = DEFAULT_RULES) -> Path:
    return cfg.data_root / "system" / f"strategy_{rules}" / "plans"


def preselect_day(panel: pd.DataFrame, d: date, rules: str = DEFAULT_RULES) -> tuple[str, pd.DataFrame]:
    day = panel[panel["trade_date"] == d]
    if day.empty:
        raise RuntimeError(f"没有 {d} 的特征,先跑 build / structure")
    stage = str(day["stage"].iloc[0])
    if rules in WF_OBJECTIVE:
        from ..evaluation.walkforward import known_before, learn_gates, outcomes, preselect_with_gates

        obj = WF_OBJECTIVE[rules]
        cutoff = d + timedelta(days=1)
        hist, market = outcomes(panel[panel["trade_date"] <= d])  # 只到 D 日:之后的价格不存在,结果未知的行被 known_before 去掉
        gates, _ = learn_gates(known_before(hist, cutoff, obj), market[market["t_date"] < cutoff], cutoff, obj)
        pool = preselect_with_gates(modes_v2.candidates(day), gates)
        pool["rule_version"] = f"WALKFORWARD_{obj.upper()}"
        _WF_GATES[d] = gates
    else:
        pool = modes_v3.preselect(modes_v2.candidates(day), rules=rules)
    pool = pool.merge(day[["ts_code", "name", "close", "board", "is_risk_warning", "sw_l2", "role", "limit_streak", "ret_1"]],
                      on="ts_code", how="left")
    return stage, pool


def write_plan(cfg: Config, d: date, panel: pd.DataFrame | None = None, replay: bool = False,
               rules: str = DEFAULT_RULES) -> Path:
    panel = load_panel(cfg) if panel is None else panel
    cal = TradingCalendar.load(calendar_path(cfg.data_root))
    t = cal.next(d)
    stage, pool = preselect_day(panel, d, rules)
    if rules in WF_OBJECTIVE:
        version = f"WALKFORWARD_{WF_OBJECTIVE[rules].upper()}"
        modes = _WF_GATES[d].get(stage, ())
        expect = f"按 {d} 及以前已收盘的结果学到的开关表" + ("" if modes else ",该阶段没有放行的模式,不开仓")
    else:
        version, table = modes_v3.RULESETS[rules]
        expect, modes = table.get(stage, ("未知阶段", ()))
    cands = []
    for r in pool.itertuples(index=False):
        band = limit_band(float(r.close), r.board, bool(r.is_risk_warning), t)
        cands.append({"order": int(r.order), "ts_code": r.ts_code, "name": r.name, "mode": r.mode, "score": round(float(r.score), 4),
                      "d_close": float(r.close), "t_up_limit": band.up, "sw_l2": r.sw_l2, "role": r.role,
                      "limit_streak": int(r.limit_streak), "d_ret": round(float(r.ret_1), 4)})
    doc = {"rule_version": version, "signal_date": str(d), "target_date": str(t), "d_stage": stage,
           "t_expectation": expect, "allowed_modes": list(modes),
           "freeze_at": (datetime.combine(d, dtime(21, 0), TZ) if replay else datetime.now(TZ)).isoformat(),
           "historical_replay": replay,
           "exit_rule": ("T+1 收盘卖;T+1 收盘在跌停价或停牌卖不出,顺延到之后第一个能卖的开盘" if rules == "wf" else "未规定(只评价当天收涨)"),
           "candidates": cands}
    out_dir = plans_root(cfg, rules) / f"{d:%Y%m%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(list(out_dir.glob("plan.v*.json"))) + 1
    out = out_dir / f"plan.v{n}.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(render_plan(doc), encoding="utf-8")
    return out


def render_plan(doc: dict) -> str:
    L = [f"# {doc['rule_version']} 交易计划:信号日 {doc['signal_date']} → 执行日 {doc['target_date']}", "",
         f"- D 日情绪阶段:**{doc['d_stage']}**;T 日预期:{doc['t_expectation']}",
         f"- 卖出:{doc.get('exit_rule', '未规定')}",
         f"- 允许开盘买入的模式:{'、'.join(f'{m} {MODE_NAMES.get(m, m)}' for m in doc['allowed_modes']) or '无(空仓)'}", ""]
    if doc["candidates"]:
        L += ["T 日 9:25 后只看下面这些票的竞价,按顺序取第一只竞价没到涨停价的,开盘买入:", "",
              "| 顺序 | 模式 | 代码 | 名称 | 行业 | 角色 | 连板 | D 日涨幅 | T 日涨停价 |", "|---|---|---|---|---|---|---|---|---|"]
        for c in doc["candidates"]:
            L.append(f"| {c['order']} | {c['mode']} | {c['ts_code']} | {c['name']} | {c['sw_l2']} | {c['role'] or '—'} | "
                     f"{c['limit_streak']} | {c['d_ret']:+.2%} | {c['t_up_limit']} |")
    return "\n".join(L) + "\n"


def latest_plan(cfg: Config, d: date, rules: str = DEFAULT_RULES) -> dict:
    files = sorted((plans_root(cfg, rules) / f"{d:%Y%m%d}").glob("plan.v*.json"), key=lambda p: int(p.stem.split(".v")[1]))
    if not files:
        raise RuntimeError(f"没有信号日 {d} 的 v3 计划,先跑 plan-v3 --date {d}")
    return json.loads(files[-1].read_text(encoding="utf-8"))


def fetch_auction(codes: list[str], t: date, now: datetime | None = None) -> list[dict]:
    """只拉预选池内股票的腾讯行情;9:26 之后、9:30 之前今开字段即 9:25 最终撮合价。"""
    import requests

    now = now or datetime.now(TZ)
    sym = [("sh" if c.endswith(".SH") else "sz") + c[:6] for c in codes]
    recs: list[dict] = []
    if not sym:
        return recs
    text = requests.get(QUOTE_URL + ",".join(sym), timeout=10).text
    for line in text.strip().split(";"):
        if "~" not in line:
            continue
        f = line.split('"')[1].split("~")
        code = f[2] + (".SH" if f[2].startswith("6") else ".SZ")
        px = float(f[5] or 0) or float(f[3] or 0)
        if code in codes and px > 0:
            recs.append({"trade_date": str(t), "ts_code": code, "price": px, "source": "tencent_qt",
                         "observed_available_at": now, "underlying_event_at": datetime.combine(t, dtime(9, 25), TZ)})
    return recs


def run_auction(cfg: Config, t: date, file: Path | None = None, n: int = 1, rules: str = DEFAULT_RULES) -> Path:
    cal = TradingCalendar.load(calendar_path(cfg.data_root))
    plan = latest_plan(cfg, cal.prev(t), rules)
    pool = pd.DataFrame(plan["candidates"])
    out_dir = plans_root(cfg, rules) / plan["signal_date"].replace("-", "")
    if pool.empty:
        out = out_dir / "auction.md"
        out.write_text(f"# {t} 竞价:D 日 {plan['d_stage']},{plan['t_expectation']},不开仓\n", encoding="utf-8")
        return out
    if file is not None:
        raw = json.loads(file.read_text(encoding="utf-8"))
        for rec in raw:
            for k in ("observed_available_at", "underlying_event_at"):
                rec[k] = pd.Timestamp(rec[k]).to_pydatetime()
    else:
        raw = fetch_auction(pool["ts_code"].tolist(), t)
    recs = [validate_auction_record(r) for r in raw]
    auction = {a.ts_code: a.price for a in recs if a.trade_date == str(t)}
    up = {c: u for c, u in zip(pool["ts_code"], pool["t_up_limit"]) if u is not None and not pd.isna(u)}
    res = modes_v3.auction_pick(pool, auction, up, n=n)
    res["gap"] = res["auction_price"] / res["d_close"] - 1
    out = out_dir / "auction.json"
    if out.exists():
        raise RuntimeError(f"{out} 已存在,竞价决策不可覆盖")
    out.write_text(res.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    L = [f"# {t} 竞价决策(信号日 {plan['signal_date']},D 日 {plan['d_stage']})", "",
         "| 顺序 | 代码 | 名称 | 模式 | 竞价 | 跳空 | 决定 |", "|---|---|---|---|---|---|---|"]
    for r in res.itertuples(index=False):
        act = "**开盘买入**" if r.buy else (r.skip_reason or "备选")
        gap = f"{r.gap:+.2%}" if pd.notna(r.gap) else "—"
        L.append(f"| {r.order} | {r.ts_code} | {r.name} | {r.mode} | {r.auction_price or '—'} | {gap} | {act} |")
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def replay_picks(panel: pd.DataFrame, days: list[date], n: int = 1, rules: str = DEFAULT_RULES) -> pd.DataFrame:
    """历史重放:T 日开盘价当作 9:25 竞价价,到涨停价 / 停牌跳过;返回每笔选中的票及当天开→收。"""
    p = panel.sort_values(["ts_code", "trade_date"])
    g = p.groupby("ts_code", sort=False)
    nxt = p.assign(t_date=g["trade_date"].shift(-1), t_open=g["open"].shift(-1), t_close=g["close"].shift(-1),
                   t_up=g["up_limit"].shift(-1))
    nxt = nxt[nxt["trade_date"].isin(days)].set_index(["trade_date", "ts_code"])
    pool = modes_v3.preselect(modes_v2.candidates(panel[panel["trade_date"].isin(days)]), rules=rules)
    rows = []
    for d, day_pool in pool.groupby("trade_date"):
        keys = [(d, c) for c in day_pool["ts_code"]]
        sub = nxt.reindex(keys)
        auction = {c: float(o) for (_, c), o in zip(keys, sub["t_open"]) if pd.notna(o) and o > 0}
        up = {c: float(u) for (_, c), u in zip(keys, sub["t_up"]) if pd.notna(u)}
        res = modes_v3.auction_pick(day_pool, auction, up, n=n)
        res = res[res["buy"]]
        for r in res.itertuples(index=False):
            x = sub.loc[(d, r.ts_code)]
            rows.append({"trade_date": d, "t_date": x["t_date"], "ts_code": r.ts_code, "mode": r.mode, "stage": r.stage,
                         "order": r.order, "t_open": x["t_open"], "t_close": x["t_close"]})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["t_oc"] = out["t_close"] / out["t_open"] - 1
        out["win"] = out["t_oc"] > SUCCESS_MARGIN
    return out


def history_v3(cfg: Config, start: date, end: date, n: int = 1, panel: pd.DataFrame | None = None,
               rules: str = DEFAULT_RULES) -> dict:
    panel = load_panel(cfg) if panel is None else panel
    cal = TradingCalendar.load(calendar_path(cfg.data_root))
    days = [x for x in cal.between(start, end) if x < end]
    picks = replay_picks(panel, days, n=n, rules=rules)
    # 对照:同期全市场(可交易、T 日非一字、非开盘涨停)开盘买入的当天收涨率
    p = panel.sort_values(["ts_code", "trade_date"])
    g = p.groupby("ts_code", sort=False)
    m = p.assign(t_open=g["open"].shift(-1), t_close=g["close"].shift(-1), t_up=g["up_limit"].shift(-1))
    m = m[m["trade_date"].isin(days) & m["eligible"].astype(bool) & (m["t_open"] > 0) & ((m["t_open"] - m["t_up"]).abs() > 1e-6)]
    base = float((m["t_close"] / m["t_open"] - 1 > SUCCESS_MARGIN).mean())
    res: dict = {"rule_version": modes_v3.RULESETS[rules][0], "range": [str(start), str(end)], "picks_per_day": n,
                 "signal_days": len(days), "market_close_up_rate": base,
                 "note": "开盘价近似 9:25 竞价价;收涨 = 收盘高于开盘 0.15% 以上"}
    if picks.empty:
        return {"metrics": res, "picks": picks}
    res.update({"days_traded": int(picks["trade_date"].nunique()), "trades": int(len(picks)),
                "close_up_rate": float(picks["win"].mean()),
                "avg_open_to_close_net": float(picks["t_oc"].mean() - SUCCESS_MARGIN),
                "by_stage": {s: {"trades": int(len(x)), "close_up_rate": float(x["win"].mean())} for s, x in picks.groupby("stage")},
                "by_mode": {s: {"trades": int(len(x)), "close_up_rate": float(x["win"].mean())} for s, x in picks.groupby("mode")}})
    wins = picks["win"].astype(float).to_numpy()
    rng = np.random.default_rng(7)
    boot = [rng.choice(wins, len(wins)).mean() for _ in range(2000)]
    res["close_up_rate_ci95"] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    return {"metrics": res, "picks": picks}
