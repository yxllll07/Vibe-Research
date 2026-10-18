from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from ashare_lab.data.raw_store import RawIntegrityError, RawStore
from ashare_lab.data.securities import classify, is_delisting_period, is_listing_day_marker, is_risk_warning
from ashare_lab.features.events import add_limit_events
from ashare_lab.features.market import market_features
from ashare_lab.labels import build_labels
from ashare_lab.rules.fees import FeeModel
from ashare_lab.rules.limits import limit_band
from ashare_lab.universe.eligibility import build_security_master, eligible_mask, listed_trading_days

from .conftest import bar, walk, weekdays


def test_limit_rounding_half_up():
    assert limit_band(10.05, "main", False, date(2025, 1, 2)).up == 11.06  # 11.055 → 11.06
    assert limit_band(10.05, "main", False, date(2025, 1, 2)).down == 9.05  # 9.045 → 9.05
    assert limit_band(10.0, "chinext", False, date(2025, 1, 2)).up == 12.0
    assert not limit_band(10.0, "main", False, date(2025, 1, 2), listed_trading_days=5).has_limit
    assert limit_band(10.0, "main", False, date(2025, 1, 2), listed_trading_days=6).up == 11.0
    # 主板全面注册制(2023-04-10)之前:首日特殊笼子(无常规涨跌停),第 2 天起 ±10%
    assert not limit_band(10.0, "main", False, date(2022, 6, 1), listed_trading_days=1).has_limit
    assert limit_band(10.0, "main", False, date(2022, 6, 1), listed_trading_days=2).up == 11.0


def test_delisting_period_first_day_has_no_limit():
    days = weekdays(date(2025, 6, 2), 3)
    rows = [bar(days[0], "600001.SH", 1.0, 0.95, name="*ST测试"), bar(days[1], "600001.SH", 0.95, 0.30, name="退市测试"),
            bar(days[2], "600001.SH", 0.30, 0.27, name="退市测试")]
    ev = add_limit_events(pd.DataFrame(rows)).set_index("trade_date")
    assert pd.isna(ev.loc[days[1], "up_limit"])  # 整理期首日无涨跌幅
    assert ev.loc[days[2], "down_limit"] == 0.27  # 之后恢复 ±10%


def test_fees_by_trade_date():
    fm = FeeModel()
    before, after = fm.cost(date(2023, 8, 25), "sell", 100_000), fm.cost(date(2023, 8, 28), "sell", 100_000)
    assert before - after == pytest.approx(50 + 100_000 * (0.0000487 - 0.0000341), abs=0.01)  # 印花税减半
    assert fm.cost(date(2025, 1, 2), "buy", 1_000) == pytest.approx(5.0 + 1_000 * 0.0000441, abs=0.01)  # 最低 5 元佣金


def test_raw_store_no_overwrite_and_integrity(tmp_path):
    s = RawStore(tmp_path)
    rec = s.put("src", "20250102", ".bin", b"abc", "http://x", 200)
    with pytest.raises(RawIntegrityError):
        s.put("src", "20250102", ".bin", b"xyz", "http://x", 200)
    assert RawStore(tmp_path).get("src", "20250102").sha256 == rec.sha256  # 续传时从清单复用
    rec.path.write_bytes(b"tampered")
    with pytest.raises(RawIntegrityError):
        RawStore(tmp_path).get("src", "20250102")


def test_security_classification():
    assert classify("SSE", "600519") == ("A", "main")
    assert classify("SSE", "000001") == ("index", "index")  # 上证指数,不是平安银行
    assert classify("SZSE", "000001") == ("A", "main")
    assert classify("SZSE", "002594") == ("A", "main")
    assert classify("SZSE", "300750")[1] == "chinext"
    assert classify("SSE", "688981")[1] == "star"
    assert is_risk_warning("*ST富润") and is_risk_warning("ST明诚") and not is_risk_warning("浦发银行")
    assert is_delisting_period("测试退") and is_delisting_period("退市海润") and not is_delisting_period("浦发银行")
    assert is_listing_day_marker("N华电") and is_listing_day_marker("C华电") and not is_listing_day_marker("NTX")


def test_ipo_first_five_days_excluded():
    days = weekdays(date(2025, 3, 3), 10)
    rows = walk(days, "600001.SH", 10.0, [0.0] * 10)
    late = walk(days[2:], "601999.SH", 8.0, [0.05] * 8)
    late[0]["name"], late[0]["listing_day_marker"] = "N新股", True
    panel = pd.DataFrame(rows + late)
    master = build_security_master(panel, days[0], days[-1])
    assert master.set_index("ts_code").loc["601999.SH", "listing_status"] == "listed_in_window"
    ld = listed_trading_days(panel, master)
    m = eligible_mask(panel, ld)
    new = panel[m & (panel.ts_code == "601999.SH")]
    assert new["trade_date"].min() == days[2 + 5]  # 上市第 6 个交易日起才进池


def test_market_breadth_denominators(panel):
    mf = market_features(add_limit_events(panel)).set_index("trade_date")
    first = mf.iloc[0]
    assert pd.isna(first["mkt_broke_rate"])  # 当天没人触板:分母 0 → NA,不是 0
    assert pd.isna(first["mkt_promo_1to2"])
    assert mf["mkt_max_height"].max() == 3


def test_labels_scoped_by_mode(panel):
    ev = add_limit_events(panel)
    lab = build_labels(ev).set_index(["signal_date", "ts_code"])
    d = sorted(panel.trade_date.unique())
    # 600003 第 5、6、7 天三连板
    assert lab.loc[(d[5], "600003.SH"), "y_B"] == True  # noqa: E712  首板 → 次日二板
    assert pd.isna(lab.loc[(d[6], "600003.SH"), "y_B"])  # 二板不属于 B
    assert lab.loc[(d[6], "600003.SH"), "y_2to3"] == True  # noqa: E712
    assert pd.isna(lab.loc[(d[5], "600003.SH"), "y_A"])  # D 日已涨停的不属于 A
    assert lab.loc[(d[4], "600003.SH"), "y_A"] == True  # noqa: E712


def test_one_word_and_unknown_fields():
    d = weekdays(date(2025, 3, 3), 1)[0]
    ev = add_limit_events(pd.DataFrame([bar(d, "600001.SH", 10.0, 11.0, open_=11.0, high=11.0, low=11.0)]))
    r = ev.iloc[0]
    assert r["one_word_up"] and r["limit_up_close"] and not r["broke_board"]
    assert pd.isna(r["open_times"]) and pd.isna(r["first_limit_time"])  # 日线不可得 → UNKNOWN,不是 0
