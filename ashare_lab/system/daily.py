"""交易系统 v1 的日常运行。

时间线(北京时间):
- D 日收盘后(通达信盘后包发布后,约 17:00~20:00):`daily --date D`
  1. 补齐日历与盘后包 → 重建特征;
  2. 判定 D 日市场状态 → 生成 T 日交易计划 → 冻结(frozen.vN.json,只增不改)+ 人读的 plan.md;
  3. 用全部已冻结的计划重放纸面账本(确定性重放,不存可变状态)→ status.md。
- T 日 9:26 后(可选):`auction --date T --file 竞价.json` 只在冻结名单内放弃 / 保留;没有竞价数据记 NO_AUCTION_DATA。
- 纸面成交口径与回测相同:T 日开盘价 + 滑点,一字 / 开盘即涨停不成交,T+1,跌停卖不出顺延。
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from ..build import build, features_dir
from ..candidates.freeze import AuctionDecision, Candidate, FrozenPool, auction_filter, write_frozen
from ..config import TZ, Config
from ..data.auction import validate_auction_record
from ..data.calendar import TradingCalendar
from ..data.ingest import calendar_path, ingest_calendar, ingest_tdx
from ..data.pit import d_postclose
from ..evaluation import proxy_backtest
from ..evaluation.stats import block_bootstrap_ci, daily_returns
from .strategy import Strategy, make_picks

CAPITAL = 1_000_000
STATE_NAMES = {0: "退潮 / 弱势", 1: "情绪高潮", 2: "温和普涨", 3: "普跌、局部活跃"}
MODE_NAMES = {"A": "0→1 首板先手", "B": "1→2 首板接力", "C": "龙头首次分歧", "D": "分歧后弱转强", "E": "趋势"}


def system_root(cfg: Config) -> Path:
    return cfg.data_root / "system" / "strategy_v1"


def pools_root(cfg: Config) -> Path:
    return system_root(cfg) / "pools"


def _load_feats(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(features_dir(cfg) / "features.parquet")


def write_plan(cfg: Config, strategy: Strategy, feats: pd.DataFrame, cal: TradingCalendar, d: date,
               replay: bool = False) -> tuple[Path, Path]:
    picks, state = make_picks(strategy, feats, [d])
    st = int(state.get(d, -1))
    t = cal.next(d)
    day = feats[feats["trade_date"] == d].set_index("ts_code")
    flags = [f"STRATEGY_{strategy.hash}", f"REGIME_STATE_{st}", "PROXY_DAILY_ONLY"] + (["HISTORICAL_REPLAY"] if replay else [])
    cands = []
    for r in picks.itertuples(index=False):
        row = day.loc[r.ts_code]
        cands.append(Candidate(
            mode=r.mode, rank=int(r.rank), ts_code=r.ts_code, score=float(r.m0_score),
            evidence_features={"name": row["name"], "close": float(row["close"]), "ret_1": float(row["ret_1"]),
                               "limit_streak": int(row["limit_streak"]), "sw_l1": row["sw_l1"],
                               "slot": r.slot if isinstance(r.slot, str) else None},
            eligible=bool(row["eligible"]), data_quality_flags=flags))
    fp = FrozenPool(model_version=f"strategy_v1:{strategy.hash}", data_snapshot_id=f"{cfg['experiment_id']}:{cfg.hash()}",
                    signal_date=d, target_date=t, signal_asof=d_postclose(d),
                    freeze_at=d_postclose(d) if replay else datetime.now(TZ), price_data_cutoff=d,
                    candidates=tuple(cands))
    frozen = write_frozen(fp, pools_root(cfg))
    md = frozen.with_name(frozen.stem + ".plan.md")
    md.write_text(render_plan(strategy, fp, st, picks, day), encoding="utf-8")
    return frozen, md


def render_plan(strategy: Strategy, fp: FrozenPool, st: int, picks: pd.DataFrame, day: pd.DataFrame) -> str:
    p = strategy.spec["params"]
    L = [f"# 交易计划 {fp.target_date}(信号日 {fp.signal_date})", "",
         "> 研究 / 纸面交易用,不是投资建议;策略 v1 在验证期为小幅亏损(见 RESULTS_eod_v1.md)。",
         f"> 策略哈希 `{strategy.hash}` · 冻结于 {fp.freeze_at:%Y-%m-%d %H:%M}(须早于 T 日 9:14:59)", "",
         f"## 市场状态:{st}({STATE_NAMES.get(st, '未知')})", ""]
    gate_on = bool(p.get("regime_gate", False))
    L.append(f"状态过滤:{'开启' if gate_on else '关闭(验证期开启反而更差,仅供参考)'}")
    L.append("")
    for m in "ABCDE":
        ok = strategy.gate_says(m, st)
        L.append(f"- {m} {MODE_NAMES[m]}:训练期该状态下{'为正' if ok else '为负'}" + ("" if ok else (" → 本次不做" if gate_on else " → 提示")))
    if st == 0:
        L.append("- ⚠️ 退潮状态下做首板接力(B)在训练 / 验证 / 2026-04~10 三段都亏损,是唯一跨期一致的周期规律。")
    first = picks[picks["rank"] == 1] if not picks.empty else picks
    L += ["", "## 买入计划(T 日开盘,冻结名单内)", ""]
    if first.empty or "slot" not in first or first["slot"].isna().all():
        L.append("**NO_TRADE**:今天没有通过状态过滤的候选。")
    else:
        L += ["| 槽位 | 模式 | 代码 | 名称 | D 日收盘 | 分数 | 仓位 | 卖出规则 |", "|---|---|---|---|---|---|---|---|"]
        for r in first.dropna(subset=["slot"]).itertuples(index=False):
            row = day.loc[r.ts_code]
            hold = p["hold_days"][r.mode]
            stop = p["stop_pct"].get(r.mode)
            exit_rule = f"T+{hold} 开盘卖" + (f";入场价 −{stop:.0%} 止损" if stop else "")
            L.append(f"| {'主动' if r.slot == 'active' else '核心'} | {r.mode} | {r.ts_code} | {row['name']} | "
                     f"{row['close']:.2f} | {r.m0_score:.3f} | 权益 {p['position_pct']:.0%} | {exit_rule} |")
    L += ["", "## 未入槽的候选(不买,仅记录;主选买不到时也不替补)", ""]
    rest = picks[picks["slot"].isna()] if not picks.empty else picks
    for r in rest.itertuples(index=False):
        why = "主动槽每天只取 A/B/D 里一只" if r.rank == 1 else "备选"
        L.append(f"- {r.mode} 第 {r.rank} 名 {r.ts_code} {day.loc[r.ts_code, 'name']} 分数 {r.m0_score:.3f}({why})")
    L += ["", "## 执行约束", "",
          "- 一字涨停 / 开盘即涨停:不追,记未成交;开盘价 + 滑点越过涨停价:不买。",
          "- T+1:当日买入不能当日卖;跌停开盘卖不出,顺延到下一个可成交日。",
          f"- 主动槽、核心槽各不超过权益 {p['slot_cap']:.0%};每模式每天最多新买 1 只;已持有的不重复买。",
          "- 9:25 竞价若可得:`auction --date T --file …` 只能在本名单内放弃,不得新增。", ""]
    return "\n".join(L)


def latest_frozen(cfg: Config) -> dict[date, dict]:
    out: dict[date, dict] = {}
    for day_dir in sorted(pools_root(cfg).glob("*")):
        versions = sorted(day_dir.glob("frozen.v*.json"), key=lambda p: int(p.stem.split(".v")[1]))
        if versions:
            d = json.loads(versions[-1].read_text(encoding="utf-8"))
            out[date.fromisoformat(d["signal_date"])] = d
    return out


def _auction_rejected(cfg: Config, target: str) -> set[str]:
    """有真实竞价数据且被放弃的代码;NO_AUCTION_DATA 不算放弃(按代理口径照常开盘买)。"""
    f = pools_root(cfg) / target.replace("-", "") / "auction_decisions.json"
    if not f.exists():
        return set()
    return {x["ts_code"] for x in json.loads(f.read_text(encoding="utf-8"))
            if not x["execution_allowed"] and x["rejection_reason"] != "NO_AUCTION_DATA"}


def picks_from_frozen(frozen: dict[date, dict], cfg: Config | None = None) -> pd.DataFrame:
    rows = []
    for sd, d in frozen.items():
        rejected = _auction_rejected(cfg, d["target_date"]) if cfg is not None else set()
        for c in d["candidates"]:
            slot = c["evidence_features"].get("slot")
            if c["rank"] == 1 and slot and c["ts_code"] not in rejected:
                rows.append({"trade_date": sd, "ts_code": c["ts_code"], "mode": c["mode"], "slot": slot, "rank": 1})
    return pd.DataFrame(rows, columns=["trade_date", "ts_code", "mode", "slot", "rank"])


def paper_status(cfg: Config, feats: pd.DataFrame, cal: TradingCalendar) -> Path:
    frozen = latest_frozen(cfg)
    out = system_root(cfg) / "paper"
    out.mkdir(parents=True, exist_ok=True)
    if not frozen:
        (out / "status.md").write_text("# 纸面账本\n\n还没有冻结计划。\n", encoding="utf-8")
        return out / "status.md"
    picks = picks_from_frozen(frozen, cfg)
    start = cal.next(min(frozen))
    last = max(feats["trade_date"])
    L = ["# 纸面账本(策略 v1)", "", f"> {proxy_backtest.LABEL}", "",
         f"- 起始资金 {CAPITAL:,.0f};第一份计划的执行日 {start};数据截至 {last}"]
    if last < start:
        L.append("- 第一份计划的执行日还没有行情,账本尚未开始。")
    else:
        res = proxy_backtest.run(picks, feats, cal, start, last, capital=CAPITAL)
        res.trades.to_csv(out / "trades.csv", index=False)
        res.equity.to_csv(out / "equity.csv", index=False)
        m = res.metrics
        L += [f"- 权益 {m.get('final_equity', CAPITAL):,.0f}(累计 {m.get('total_return', 0):.2%}),最大回撤 {m.get('max_drawdown', 0):.2%}",
              f"- 成交 {m.get('trades', 0)} 笔,胜率 {m.get('win_rate', float('nan')):.1%},未成交 {m.get('unfilled_buys', 0)}",
              f"- 未成交原因:{m.get('unfilled_by_reason', {})}", "", "## 最近成交", ""]
        if not res.trades.empty:
            L += ["```", res.trades.tail(10).to_string(index=False), "```"]
    (out / "status.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return out / "status.md"


def run_daily(cfg: Config, d: date) -> dict[str, str]:
    ingest_calendar(cfg)
    cal = TradingCalendar.load(calendar_path(cfg.data_root))
    if not cal.is_open(d):
        raise RuntimeError(f"{d} 不是交易日")
    rep = ingest_tdx(cfg, date.fromisoformat(cfg["data"]["ingest_start"]), d)
    if rep.failed or rep.missing_404:
        raise RuntimeError(f"盘后包没有补齐:{rep.to_json()}")
    build(cfg, date.fromisoformat(cfg["data"]["ingest_start"]), d)
    strategy = Strategy.load(cfg)
    feats = _load_feats(cfg)
    frozen, md = write_plan(cfg, strategy, feats, cal, d)
    status = paper_status(cfg, feats, cal)
    return {"frozen": str(frozen), "plan": str(md), "paper": str(status)}


def run_auction(cfg: Config, t: date, file: Path | None) -> Path:
    """T 日竞价二次过滤:只在冻结名单内放弃 / 保留;文件格式为竞价记录列表(见 data/auction.py 白名单)。"""
    frozen = [v for v in latest_frozen(cfg).values() if v["target_date"] == str(t)]
    if not frozen:
        raise RuntimeError(f"没有目标日为 {t} 的冻结计划")
    d = frozen[0]
    cands = tuple(Candidate(**{k: c[k] for k in ("mode", "rank", "ts_code", "score", "evidence_features",
                                                    "eligible", "forbidden_reason", "data_quality_flags")})
                  for c in d["candidates"])
    pool = FrozenPool(d["model_version"], d["data_snapshot_id"], date.fromisoformat(d["signal_date"]), t,
                      pd.Timestamp(d["signal_asof"]).to_pydatetime(), pd.Timestamp(d["freeze_at"]).to_pydatetime(),
                      date.fromisoformat(d["price_data_cutoff"]), cands)
    auctions = {}
    if file is not None:
        for rec in json.loads(file.read_text(encoding="utf-8")):
            for k in ("observed_available_at", "underlying_event_at"):
                rec[k] = pd.Timestamp(rec[k]).to_pydatetime()
            a = validate_auction_record(rec)
            auctions[a.ts_code] = a
    prev_close = {c.ts_code: float(c.evidence_features["close"]) for c in cands}
    decisions: list[AuctionDecision] = auction_filter(pool, auctions, prev_close)
    out = pools_root(cfg) / f"{t:%Y%m%d}" / "auction_decisions.json"
    if out.exists():
        raise RuntimeError(f"{out} 已存在,竞价决策不可覆盖")
    out.write_text(json.dumps([vars(x) for x in decisions], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return out


def history(cfg: Config, start: date, end: date) -> dict:
    """策略 v1 在历史区间的完整模拟(逐日用当天数据生成计划,不写冻结文件)。"""
    strategy = Strategy.load(cfg)
    feats = _load_feats(cfg)
    cal = TradingCalendar.load(calendar_path(cfg.data_root))
    days = [x for x in cal.between(start, end) if x < end]
    pool, state = make_picks(strategy, feats, days)
    picks = pool.dropna(subset=["slot"])
    res = proxy_backtest.run(picks, feats, cal, cal.next(days[0]), end, capital=CAPITAL)
    m = res.metrics
    m["daily_return_bootstrap"] = block_bootstrap_ci(daily_returns(res.equity, CAPITAL))
    m["state_days"] = state.value_counts().sort_index().to_dict()
    m["no_trade_days_by_gate"] = int(len(days) - picks["trade_date"].nunique()) if not picks.empty else len(days)
    return {"metrics": m, "trades": res.trades, "equity": res.equity}
