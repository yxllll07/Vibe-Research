"""v3:D 日阶段决定 T 日能做的模式;T 日竞价只在预选池内筛,到涨停价 / 停牌跳过。"""
from __future__ import annotations

from datetime import date

import pandas as pd

from ashare_lab.candidates import modes_v3


def _cands():
    d1, d2 = date(2026, 1, 5), date(2026, 1, 6)
    return pd.DataFrame([
        {"trade_date": d1, "ts_code": "000001.SZ", "mode": "A", "score": 0.9, "stage": "修复", "gate_ok": True},
        {"trade_date": d1, "ts_code": "000002.SZ", "mode": "B", "score": 0.5, "stage": "修复", "gate_ok": True},
        {"trade_date": d1, "ts_code": "000002.SZ", "mode": "A", "score": 0.99, "stage": "修复", "gate_ok": True},
        {"trade_date": d1, "ts_code": "000003.SZ", "mode": "E", "score": 0.99, "stage": "修复", "gate_ok": True},
        {"trade_date": d2, "ts_code": "000004.SZ", "mode": "B", "score": 0.9, "stage": "高潮", "gate_ok": True},
    ])


def test_climax_day_opens_nothing_and_b_beats_a():
    pool = modes_v3.preselect(_cands())
    assert set(pool["trade_date"]) == {date(2026, 1, 5)}  # 高潮次日不接力
    assert pool["ts_code"].tolist() == ["000002.SZ", "000001.SZ"]  # B 优先,同一只票只留一次,E 不在修复日允许范围
    assert pool.loc[0, "mode"] == "B"


def test_every_stage_has_expectation():
    from ashare_lab.structure.sentiment import STAGES

    assert set(STAGES) == set(modes_v3.EXPECT)
    for s in ("发酵", "高潮", "分歧"):
        assert modes_v3.allowed_modes(s) == ()


def test_auction_pick_skips_limit_up_and_suspended():
    pool = pd.DataFrame({"order": [1, 2, 3, 4], "ts_code": ["a", "b", "c", "d"]})
    res = modes_v3.auction_pick(pool, {"a": 11.0, "c": 10.5, "d": 10.2}, {"a": 11.0, "b": 11.0, "c": 11.0, "d": 11.0}, n=1)
    assert res.set_index("ts_code")["buy"].to_dict() == {"a": False, "b": False, "c": True, "d": False}
    assert "涨停" in res.loc[0, "skip_reason"] and "停牌" in res.loc[1, "skip_reason"]
