"""数据质量报告:覆盖率、日历一致性、价格带越界、前收连续性、股票池结构、与东财涨停池交叉核对。"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from ..config import REPO_ROOT, Config
from ..features.events import add_limit_events
from ..universe.eligibility import build_security_master, listed_trading_days
from .calendar import TradingCalendar
from .ingest import calendar_path
from .panel import load_panel
from .tdx import normalized_path

_SCRIPTS = REPO_ROOT / ".agents" / "skills" / "data-access" / "scripts"


def em_zt_pool_codes(d: date) -> set[str] | None:
    """东财涨停池(只保留约一个月);取不到返回 None,不当成 0 只。"""
    out = subprocess.run([sys.executable, str(_SCRIPTS / "fetch_endpoint.py"), "--endpoint", "em_zt_pool",
                          "--args", json.dumps({"date": d.strftime("%Y%m%d")})],
                         capture_output=True, text=True, timeout=120)
    try:
        res = json.loads(out.stdout)
    except json.JSONDecodeError:
        return None
    if res.get("status") not in ("ok", "partial"):
        return None
    return {e["record_key"][:6] for e in res["evidence"] if e.get("field") == "pool_limit_days" and e.get("record_key")}


def run_quality(cfg: Config, start: date, end: date, crosscheck: list[date]) -> Path:
    root = cfg.data_root
    cal = TradingCalendar.load(calendar_path(root))
    days = cal.between(start, end)
    present = [d for d in days if normalized_path(root, d).exists()]
    missing = [str(d) for d in days if d not in set(present)]
    panel = load_panel(root, start, end)
    master = build_security_master(panel, present[0], present[-1])
    ld = listed_trading_days(panel, master)
    ev = add_limit_events(panel, ld)
    main = ev[(ev["board"] == "main")]
    traded = main[~main["is_suspended"]]

    # 价格带越界:有涨跌幅限制却收在带外 → 数据或规则问题,逐条列出
    has_band = traded["up_limit"].notna()
    out_band = traded[has_band & ((traded["close"] > traded["up_limit"] + 1e-6) |
                                  (traded["close"] < traded["down_limit"] - 1e-6))]
    # 前收连续性:今日前收 ≠ 昨日收盘 → 除权除息(正常)或数据断档
    s = main.sort_values(["ts_code", "trade_date"])
    prev_close_gap = (s["prev_close"] - s.groupby("ts_code")["close"].shift(1)).abs() > 0.005
    by_month = (main.assign(month=pd.to_datetime(main["trade_date"]).dt.strftime("%Y-%m"))
                .groupby("month").agg(days=("trade_date", "nunique"), rows=("ts_code", "size")))
    cal_month = pd.Series([d.strftime("%Y-%m") for d in days]).value_counts().sort_index()
    by_month["calendar_days"] = cal_month
    by_month["coverage"] = (by_month["days"] / by_month["calendar_days"]).round(4)

    daily = traded.groupby("trade_date").agg(
        main_traded=("ts_code", "size"), limit_up=("limit_up_close", "sum"), broke=("broke_board", "sum"),
        limit_down=("limit_down_close", "sum"), st=("is_risk_warning", "sum"))
    cross = []
    for d in crosscheck:
        em = em_zt_pool_codes(d)
        day = traded[(traded["trade_date"] == d) & ~traded["is_risk_warning"]]
        mine = set(day.loc[day["limit_up_close"], "code"])
        if em is None:
            cross.append({"date": str(d), "status": "EM_UNAVAILABLE", "tdx_rebuilt": len(mine)})
            continue
        em_main = {c for c in em if c in set(main["code"])}
        cross.append({"date": str(d), "status": "ok", "tdx_rebuilt": len(mine), "em_main": len(em_main),
                      "both": len(mine & em_main), "only_tdx": sorted(mine - em_main), "only_em": sorted(em_main - mine)})

    summary = {
        "range": [str(start), str(end)], "calendar_trading_days": len(days), "present_days": len(present),
        "missing_days": missing, "coverage": round(len(present) / len(days), 4) if days else None,
        "main_board_securities": int(master[master.board == "main"].shape[0]),
        "listing_status": master[master.board == "main"]["listing_status"].value_counts().to_dict(),
        "delisted_in_window_main": int(master[(master.board == "main") & master.delisted_by_window_end].shape[0]),
        "unverified_listing_main": master[(master.board == "main") &
                                          (master.listing_status == "UNVERIFIED_LISTING")]["ts_code"].tolist()[:50],
        "out_of_band_rows": int(len(out_band)),
        "out_of_band_sample": out_band[["trade_date", "ts_code", "name", "prev_close", "close", "up_limit",
                                        "down_limit"]].head(20).astype(str).to_dict("records"),
        "prev_close_discontinuities_main": int(prev_close_gap.sum()),
        "limit_up_per_day": {"mean": round(float(daily["limit_up"].mean()), 1), "max": int(daily["limit_up"].max()),
                             "min": int(daily["limit_up"].min())},
        "crosscheck_em_zt_pool": cross,
        "unknown_fields": ["open_times", "first_limit_time", "turnover_rate", "circ_mv", "total_mv"],
    }
    rep = root / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    tag = f"{start:%Y%m%d}_{end:%Y%m%d}"
    (rep / f"quality_{tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    by_month.to_csv(rep / f"coverage_by_month_{tag}.csv")
    daily.to_csv(rep / f"daily_counts_{tag}.csv")
    return rep / f"quality_{tag}.json"
