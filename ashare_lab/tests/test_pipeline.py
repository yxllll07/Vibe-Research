"""候选池、槽位与代理回测(合成夹具)。"""
from __future__ import annotations

from datetime import date

import pandas as pd

from ashare_lab.candidates.modes import pick_slots, top_k
from ashare_lab.data.calendar import TradingCalendar
from ashare_lab.evaluation import proxy_backtest
from ashare_lab.features.events import add_limit_events
from ashare_lab.labels import build_labels
from ashare_lab.evaluation.models import add_trade_targets

from .conftest import bar, weekdays


def test_top_k_and_slots():
    d = date(2025, 3, 3)
    c = pd.DataFrame([
        {"trade_date": d, "ts_code": f"60000{i}.SH", "mode": m, "s": s}
        for i, (m, s) in enumerate([("A", 0.6), ("A", 0.9), ("A", 0.1), ("B", 0.7), ("D", 0.95), ("C", 0.5), ("E", 0.4)])
    ])
    pool = top_k(c, "s", k=2)
    assert pool.groupby("mode").size().max() <= 2
    assert set(pool[pool["mode"] == "A"]["s"]) == {0.9, 0.6}
    slots = pick_slots(pool, "s")
    act = slots[slots["slot"] == "active"]
    assert len(act) == 1 and act.iloc[0]["mode"] == "D"  # A/B/D 主选里分数最高的一只
    assert set(slots[slots["slot"] == "core"]["mode"]) == {"C", "E"}


def _feats(rows):
    return add_limit_events(pd.DataFrame(rows))


def test_proxy_backtest_one_word_not_filled_and_t1_exit():
    days = weekdays(date(2025, 3, 3), 6)
    cal = TradingCalendar(days)
    rows = []
    # 600001:D=day0 入选 B;T=day1 一字涨停 → 买不到
    rows += [bar(days[0], "600001.SH", 9.09, 10.0), bar(days[1], "600001.SH", 10.0, 11.0, open_=11.0, high=11.0, low=11.0)]
    rows += [bar(d, "600001.SH", 11.0, 11.0) for d in days[2:]]
    # 600002:D=day0 入选 A;T=day1 开盘 10 买入,day2 开盘 10.5 卖出
    rows += [bar(days[0], "600002.SH", 10.0, 10.0), bar(days[1], "600002.SH", 10.0, 10.4, open_=10.0),
             bar(days[2], "600002.SH", 10.4, 10.6, open_=10.5)] + [bar(d, "600002.SH", 10.6, 10.6) for d in days[3:]]
    feats = _feats(rows)
    picks = pd.DataFrame([
        {"trade_date": days[0], "ts_code": "600001.SH", "mode": "B", "slot": "active", "rank": 1},
        {"trade_date": days[0], "ts_code": "600002.SH", "mode": "A", "slot": "core", "rank": 1},
    ])
    picks.loc[1, "slot"] = "active"
    res = proxy_backtest.run(picks, feats, cal, days[0], days[-1], capital=1_000_000)
    assert res.metrics["unfilled_by_reason"].get("ONE_WORD_LIMIT_UP_NO_FILL") == 1
    t = res.trades.iloc[0]
    assert t["ts_code"] == "600002.SH" and t["entry_date"] == days[1] and t["exit_date"] == days[2]
    assert abs(t["entry_price"] - 10.01) < 1e-9 and abs(t["exit_price"] - 10.49) < 1e-9
    assert "不代表" in res.metrics["label"]


def test_trade_target_open_to_next_open():
    days = weekdays(date(2025, 3, 3), 4)
    rows = [bar(days[0], "600001.SH", 10.0, 10.0), bar(days[1], "600001.SH", 10.0, 10.4, open_=10.2),
            bar(days[2], "600001.SH", 10.4, 10.0, open_=10.71), bar(days[3], "600001.SH", 10.0, 10.0)]
    ev = _feats(rows)
    lab = add_trade_targets(build_labels(ev, (1,)), ev).set_index("signal_date")
    assert abs(lab.loc[days[0], "r_open_next_open"] - (10.71 / 10.2 - 1)) < 1e-9
