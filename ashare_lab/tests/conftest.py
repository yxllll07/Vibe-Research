"""合成夹具。只用于单元测试,结果绝不进入真实数据报告。"""
from __future__ import annotations

import sys
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ashare_lab.config import TZ  # noqa: E402
from ashare_lab.data.calendar import TradingCalendar  # noqa: E402
from ashare_lab.rules.limits import limit_band  # noqa: E402


def weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def bar(d: date, ts: str, prev_close: float, close: float, *, open_: float | None = None, high: float | None = None,
        low: float | None = None, name: str = "测试股份", board: str = "main", exchange: str = "SSE",
        volume: int = 1_000_000) -> dict:
    o = prev_close if open_ is None else open_
    return {
        "trade_date": d, "ts_code": ts, "exchange": exchange, "code": ts[:6], "name": name,
        "security_type": "A", "board": board, "prev_close": prev_close, "open": o,
        "high": max(o, close) if high is None else high, "low": min(o, close) if low is None else low,
        "close": close, "volume_shares": volume, "amount_cny": volume * close,
        "is_risk_warning": "ST" in name, "is_delisting_period": name.endswith("退"),
        "listing_day_marker": name[:1] in ("N", "C") and not name[1:2].isascii(),
        "is_suspended": volume == 0,
        "event_time": datetime.combine(d, dtime(15), TZ), "available_at": datetime.combine(d, dtime(20), TZ),
        "source": "SYNTHETIC",
    }


def walk(days: list[date], ts: str, start_price: float, rets: list[float], **kw) -> list[dict]:
    rows, pc = [], start_price
    for d, r in zip(days, rets):
        up = limit_band(pc, kw.get("board", "main"), "ST" in kw.get("name", ""), d).up
        close = up if r == "LU" else round(pc * (1 + r), 2)
        rows.append(bar(d, ts, pc, close, **kw))
        pc = close
    return rows


@pytest.fixture
def days() -> list[date]:
    return weekdays(date(2025, 3, 3), 40)


@pytest.fixture
def cal(days) -> TradingCalendar:
    return TradingCalendar(days)


@pytest.fixture
def panel(days) -> pd.DataFrame:
    rows: list[dict] = []
    rows += walk(days, "600001.SH", 10.0, [0.01] * 40)
    # 600002:第 10~19 天戴帽 ST
    for i, r in enumerate(walk(days, "600002.SH", 8.0, [0.0] * 40)):
        if 10 <= i < 20:
            r["name"], r["is_risk_warning"] = "ST测试", True
        rows.append(r)
    # 600003:第 5 天起三连板,之后回落
    rows += walk(days, "600003.SH", 5.0, [0.0] * 5 + ["LU", "LU", "LU"] + [-0.03] * 32)
    # 300001:创业板,不在主板股票池
    rows += walk(days, "300001.SZ", 20.0, [0.0] * 40, board="chinext", exchange="SZSE")
    # 600004:第 0~14 天正常,15~19 天退市整理,第 20 天起从快照消失(已退市)
    for i, r in enumerate(walk(days[:20], "600004.SH", 3.0, [-0.02] * 20)):
        if i >= 15:
            r["name"], r["is_delisting_period"] = "测试退", True
        rows.append(r)
    return pd.DataFrame(rows)
