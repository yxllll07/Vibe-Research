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


def test_v4_only_adds_b_on_divergence_and_keeps_v3_frozen():
    from ashare_lab.structure.sentiment import STAGES

    for s in STAGES:
        v3, v4 = modes_v3.allowed_modes(s, "v3"), modes_v3.allowed_modes(s, "v4")
        assert v4 == (("B",) if s == "分歧" else v3)
    assert modes_v3.allowed_modes("分歧", "v3") == ()


def test_walkforward_refuses_future_outcomes():
    import pytest

    from ashare_lab.data.pit import LeakageError
    from ashare_lab.evaluation.walkforward import learn_gates

    hist = pd.DataFrame({"trade_date": [date(2026, 1, 5)], "t_date": [date(2026, 1, 6)], "ts_code": ["a"], "mode": ["B"],
                         "stage": ["震荡"], "score": [1.0], "buyable": [True], "win": [True]})
    market = pd.DataFrame({"trade_date": [date(2026, 1, 5)], "t_date": [date(2026, 1, 6)], "mkt_win": [0.5]})
    learn_gates(hist, market, cutoff=date(2026, 1, 7), objective="close_up")
    with pytest.raises(LeakageError):
        learn_gates(hist, market, cutoff=date(2026, 1, 6), objective="close_up")  # T 日 = 截止日,当时还没收盘
    # 可兑现口径:T+1 收盘才知道结果,卖出日 = 截止日也算未来
    real = hist.assign(r_close1=[0.01], x_date=[date(2026, 1, 7)])
    learn_gates(real, market, cutoff=date(2026, 1, 8))
    with pytest.raises(LeakageError):
        learn_gates(real, market, cutoff=date(2026, 1, 7))
    with pytest.raises(LeakageError):
        learn_gates(real.assign(x_date=[None]), market, cutoff=date(2026, 1, 8))  # 卖出日未知


def test_relay_score_skips_limit_up_and_suspended(tmp_path):
    import lightgbm as lgb
    import numpy as np

    from ashare_lab.system import relay

    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(rng.normal(size=(n, len(relay.FEATURES))), columns=relay.FEATURES)
    m = lgb.LGBMClassifier(n_estimators=5, verbose=-1).fit(X, (X["gap"] < 0).astype(int))
    m.booster_.save_model(str(tmp_path / "m.txt"))
    snap = pd.DataFrame(rng.normal(size=(3, len(relay.D_FEATURES))), columns=relay.D_FEATURES)
    snap = snap.assign(ts_code=["a", "b", "c"], name=["A", "B", "C"], board="main", is_risk_warning=False, close=10.0)
    # c 当天除权,参考价 9.5:竞价 9.9 是高开 4.2%,不是低开 1%
    top = relay.score_day(snap, {"a": 11.0, "c": 9.9}, {"c": 9.5}, date(2026, 10, 19), tmp_path / "m.txt")
    assert top["ts_code"].tolist() == ["c"]  # a 竞价到涨停价、b 无竞价,都跳过
    assert abs(top.loc[0, "gap"] - (9.9 / 9.5 - 1)) < 1e-9
