"""研究方案第 10 章的 12 条红线,全部用合成夹具。"""
from __future__ import annotations

import json
from datetime import date, datetime, time as dtime

import pandas as pd
import pytest

from ashare_lab.candidates.freeze import (Candidate, FreezeError, FrozenPool, auction_filter, write_frozen)
from ashare_lab.config import TZ
from ashare_lab.data.auction import AuctionSchemaError, validate_auction_record
from ashare_lab.data.pit import AsOfView, LeakageError, auction_decision, d_postclose
from ashare_lab.evaluation.split import Period, TrainOnlyScaler, assign_split
from ashare_lab.features.events import add_limit_events
from ashare_lab.features.stock import STOCK_FEATURES, add_stock_features
from ashare_lab.labels import build_labels
from ashare_lab.rules.execution import Bar, Broker
from ashare_lab.rules.limits import limit_band
from ashare_lab.universe.eligibility import build_security_master, eligible_mask
from ashare_lab.universe.sector import members_asof

from .conftest import bar, walk


def _bar(d, prev_close, o, h, l, c, vol=1_000_000, board="main", st=False):
    return Bar(d, o, h, l, c, prev_close, vol, limit_band(prev_close, board, st, d))


# 1
def test_market_and_st_filter_by_date(panel, days):
    p = panel.set_index(["trade_date", "ts_code"])
    m = eligible_mask(panel)
    elig = set(map(tuple, panel.loc[m, ["trade_date", "ts_code"]].itertuples(index=False)))
    assert (days[9], "600002.SH") in elig
    assert all((days[i], "600002.SH") not in elig for i in range(10, 20))  # ST 期间按当日简称排除
    assert (days[20], "600002.SH") in elig  # 摘帽后恢复
    assert not any(ts == "300001.SZ" for _, ts in elig)  # 创业板不进池
    assert p.loc[(days[12], "600002.SH"), "is_risk_warning"]
    # 主板 ST 涨跌幅按日期分段:2026-07-06 前 5%,之后 10%
    assert limit_band(10.0, "main", True, date(2026, 7, 3)).up == 10.50
    assert limit_band(10.0, "main", True, date(2026, 7, 6)).up == 11.00


# 2
def test_delisted_stock_in_past_universe(panel, days):
    m = eligible_mask(panel)
    gone = panel[m & (panel["ts_code"] == "600004.SH")]
    assert set(gone["trade_date"]) == set(days[:15])  # 退市前正常交易日都在样本里
    master = build_security_master(panel, days[0], days[-1]).set_index("ts_code")
    assert master.loc["600004.SH", "delisted_by_window_end"]
    assert master.loc["600004.SH", "last_seen"] == days[19]
    assert not master.loc["600001.SH", "delisted_by_window_end"]


# 3
def test_availability_guard(panel, days):
    ev = add_limit_events(panel)
    view = AsOfView({"daily_bar": panel, "limit_event": ev})
    D, T = days[10], days[11]
    view.rows_for_date("daily_bar", D, d_postclose(D))  # D 日盘后可用
    for table in ("daily_bar", "limit_event"):  # T 日收盘、T 日涨停榜在 T 日 9:26 都不可见
        with pytest.raises(LeakageError):
            view.rows_for_date(table, T, auction_decision(T))
    # D 日 15:30(数据尚未发布)也不能用 D 日日线
    with pytest.raises(LeakageError):
        view.rows_for_date("daily_bar", D, datetime.combine(D, dtime(15, 30), TZ))
    hist = view.history("daily_bar", auction_decision(T))
    assert hist["trade_date"].max() == D


# 4
def test_auction_final_only():
    t = date(2026, 10, 19)
    ok = {"trade_date": "2026-10-19", "ts_code": "600001.SH", "price": 10.2, "source": "x",
          "underlying_event_at": datetime.combine(t, dtime(9, 25), TZ),
          "observed_available_at": datetime.combine(t, dtime(9, 26, 30), TZ)}
    validate_auction_record(ok)
    for bad_field in ("g20", "g24", "delta_gap", "price_0920"):
        with pytest.raises(AuctionSchemaError):
            validate_auction_record({**ok, bad_field: 1.0})
    with pytest.raises(AuctionSchemaError):
        validate_auction_record({**ok, "underlying_event_at": datetime.combine(t, dtime(9, 20), TZ)})
    with pytest.raises(AuctionSchemaError):
        validate_auction_record({**ok, "observed_available_at": datetime.combine(t, dtime(9, 25, 10), TZ)})


def _pool(D, T, freeze_at=None, codes=("600001.SH", "600003.SH")):
    cands = tuple(Candidate(mode="B", rank=i + 1, ts_code=c, score=0.5 - i * 0.1, evidence_features={"x": 1})
                  for i, c in enumerate(codes))
    return FrozenPool("m0", "snap", D, T, d_postclose(D), freeze_at or d_postclose(D), D, cands)


# 5
def test_freeze_immutable(tmp_path, days):
    D, T = days[10], days[11]
    p1 = write_frozen(_pool(D, T), tmp_path)
    before = p1.read_bytes()
    p2 = write_frozen(_pool(D, T, codes=("600001.SH",)), tmp_path)
    assert p1.read_bytes() == before and p2.name == "frozen.v2.json"
    assert json.loads(p2.read_text())["previous_version_sha256"]
    with pytest.raises(FreezeError):  # 9:15 之后不能再冻结
        write_frozen(_pool(D, T, freeze_at=datetime.combine(T, dtime(9, 20), TZ)), tmp_path)
    pool = _pool(D, T)
    with pytest.raises(FreezeError):  # 盘前不得临时新增
        auction_filter(pool, {}, {}, requested=["600001.SH", "600999.SH"])
    dec = auction_filter(pool, {}, {})
    assert all(x.rejection_reason == "NO_AUCTION_DATA" and x.auction_price is None for x in dec)


# 6
def test_tplus1_sell_restriction(days):
    b = Broker(cash=100_000)
    d0, d1 = days[0], days[1]
    assert b.buy_at_open("600001.SH", _bar(d0, 10, 10, 10.3, 9.9, 10.2), 50_000)
    assert b.sell_at_open("600001.SH", _bar(d0, 10, 10, 10.3, 9.9, 10.2)) is None
    assert b.rejections[-1].reason == "T_PLUS_ONE"
    assert b.stop_loss("600001.SH", _bar(d0, 10, 10, 10.3, 9.0, 10.2), 9.5) is None
    assert b.sell_at_open("600001.SH", _bar(d1, 10.2, 10.3, 10.5, 10.1, 10.4)) is not None


# 7
def test_oneword_up_limit_no_fill(days):
    b = Broker(cash=100_000)
    d = days[0]
    assert b.buy_at_open("600003.SH", _bar(d, 10, 11, 11, 11, 11), 50_000) is None
    assert b.rejections[-1].reason == "ONE_WORD_LIMIT_UP_NO_FILL"
    assert b.buy_at_open("600003.SH", _bar(d, 10, 11, 11, 10.6, 11), 50_000) is None  # 开盘即封板也不算买到
    assert b.buy_at_open("600003.SH", _bar(d, 10, 10.99, 11, 10.8, 11), 50_000) is None  # 滑点越过涨停价
    assert b.cash == 100_000 and not b.fills


# 8
def test_locked_limit_down_no_exit(days):
    b = Broker(cash=100_000)
    b.buy_at_open("600001.SH", _bar(days[0], 10, 10, 10.1, 9.9, 10), 50_000)
    assert b.stop_loss("600001.SH", _bar(days[1], 10, 9, 9, 9, 9), 9.5) is None  # 一字跌停卖不出
    assert b.rejections[-1].reason == "LOCKED_LIMIT_DOWN_STOP_DEFERRED"
    held = b.shares_held("600001.SH")
    assert held > 0
    f = b.stop_loss("600001.SH", _bar(days[2], 9, 8.5, 8.9, 8.2, 8.6), 9.5)  # 次日跳空:按开盘价,不是止损价
    assert f is not None and f.shares == held and f.price < 9.5 and abs(f.price - 8.5 * 0.999) < 0.011


# 9
def test_split_purge_and_scaler(days, cal):
    periods = [Period("train", days[0], days[19]), Period("validation", days[20], days[39])]
    s = pd.Series(days)
    split = assign_split(s, periods, cal, max_horizon=5, embargo=3)
    train_days = s[split == "train"]
    assert train_days.max() == days[20 - 5 - 1]  # 标签窗口不得跨进验证期
    assert s[split == "validation"].min() == days[23]  # 验证期开头 embargo
    X = pd.DataFrame({"f": range(40)}, dtype=float)
    sc = TrainOnlyScaler(train_end=days[19])
    with pytest.raises(ValueError):
        sc.fit(X, s)  # 混入验证期样本
    sc.fit(X[split == "train"], train_days)
    assert sc.fit_range[1] <= days[19]


# 10
def test_pit_sector_membership():
    mem = pd.DataFrame([
        {"ts_code": "600001.SH", "sector_id": "SW_X", "provider": "sw", "effective_from": date(2020, 1, 1),
         "effective_to": None, "first_known_at": datetime(2020, 1, 1, tzinfo=TZ)},
        # 2026 年才抓到的概念成分,不得回填到 2025
        {"ts_code": "600002.SH", "sector_id": "CONCEPT_LOW_ALT", "provider": "concept", "effective_from": date(2024, 1, 1),
         "effective_to": None, "first_known_at": datetime(2026, 10, 17, tzinfo=TZ)},
        {"ts_code": "600003.SH", "sector_id": "SW_X", "provider": "sw", "effective_from": date(2020, 1, 1),
         "effective_to": date(2025, 6, 1), "first_known_at": datetime(2020, 1, 1, tzinfo=TZ)},
    ])
    d = date(2025, 8, 1)
    got = set(members_asof(mem, d, d_postclose(d))["ts_code"])
    assert got == {"600001.SH"}
    later = date(2026, 10, 19)
    assert "600002.SH" in set(members_asof(mem, later, d_postclose(later))["ts_code"])
    with pytest.raises(ValueError):
        members_asof(mem.assign(first_known_at=None), d, d_postclose(d))


# 11
def test_price_adjustment_vs_limit_raw(days):
    d0, d1, d2 = days[0], days[1], days[2]
    # 10 送 10:除权日交易所前收 = 10.00(不是上一日原始收盘 20.00)
    rows = [bar(d0, "600001.SH", 19.8, 20.0), bar(d1, "600001.SH", 10.0, 10.5), bar(d2, "600001.SH", 10.5, 10.6)]
    ev = add_limit_events(pd.DataFrame(rows)).set_index("trade_date")
    assert ev.loc[d1, "up_limit"] == 11.00  # 以除权参考价为基准,不是 22.00
    feats = add_stock_features(ev.reset_index()).set_index("trade_date")
    assert abs(feats.loc[d1, "ret_1"] - 0.05) < 1e-9  # 不是 -47.5%
    lab = build_labels(ev.reset_index(), horizons=(1,)).set_index("signal_date")
    assert abs(lab.loc[d0, "r_h1"] - (10.5 / 10.0 - 1)) < 1e-9  # T 开盘 10.0 → 收盘 10.5


# 12
def test_no_forward_core_role(panel, days):
    D = days[15]
    base = add_stock_features(add_limit_events(panel))
    # 改写 D 之后的未来:600001 之后连拉涨停成「妖股」
    fut = panel.copy()
    later = fut["trade_date"] > D
    mask = later & (fut["ts_code"] == "600001.SH")
    fut.loc[mask, "close"] = [limit_band(pc, "main", False, d).up for pc, d in
                              zip(fut.loc[mask, "prev_close"], fut.loc[mask, "trade_date"])]
    fut.loc[mask, "high"] = fut.loc[mask, "close"]
    alt = add_stock_features(add_limit_events(fut))
    cols = ["trade_date", "ts_code"] + STOCK_FEATURES
    a = base[base["trade_date"] <= D][cols].sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    b = alt[alt["trade_date"] <= D][cols].sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)  # D 日及以前的特征(含角色类:连板 / 涨停计数)不因未来改变
    la = build_labels(add_limit_events(panel)).set_index(["signal_date", "ts_code"])
    lb = build_labels(add_limit_events(fut)).set_index(["signal_date", "ts_code"])
    assert la.loc[(D, "600001.SH"), "y_A"] != lb.loc[(D, "600001.SH"), "y_A"]  # 未来只体现在标签表
