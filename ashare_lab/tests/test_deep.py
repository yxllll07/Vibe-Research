"""M2 / M3 的输入窗口只含 D 日及以前;改写未来行情不改变 D 日的输入张量。"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ashare_lab.evaluation import deep  # noqa: E402
from ashare_lab.features.events import add_limit_events  # noqa: E402
from ashare_lab.features.market import market_features  # noqa: E402
from ashare_lab.features.sector import add_sector_features  # noqa: E402
from ashare_lab.features.stock import add_stock_features  # noqa: E402
from ashare_lab.universe.eligibility import eligible_mask  # noqa: E402


def _feats(panel: pd.DataFrame) -> pd.DataFrame:
    f = add_stock_features(add_limit_events(panel))
    f["sw_l1"] = np.where(f["ts_code"].isin(["600001.SH", "600003.SH"]), "SW1_11", "SW1_22")
    f = add_sector_features(f).merge(market_features(add_limit_events(panel)), on="trade_date", how="left")
    f["eligible"] = eligible_mask(f)
    return f


def test_deep_inputs_do_not_see_future(panel, days):
    D = days[25]
    base = _feats(panel)
    fut = panel.copy()
    later = fut["trade_date"] > D
    fut.loc[later, ["close", "high"]] = fut.loc[later, ["close", "high"]] * 1.05
    alt = _feats(fut)
    rows = pd.DataFrame({"trade_date": [D, D], "ts_code": ["600001.SH", "600003.SH"], "mode": ["E", "B"]})
    # 标准化统计量只用 ≤ D 的日期拟合,于是 D 日的全部输入必须逐位相同
    Ta, Tb = deep.build_tensors(base, D), deep.build_tensors(alt, D)
    ba, bb = deep.make_batch(Ta, rows), deep.make_batch(Tb, rows)
    assert ba["cand_seq"].shape == (2, deep.L, len(deep.SEQ_CHANNELS) + 1)
    for k in ba:
        assert torch.equal(ba[k], bb[k]), k


def test_window_excludes_later_dates(panel, days):
    T = deep.build_tensors(_feats(panel), days[-1])
    T.seq[:] = 0
    s = T.codes.index("600001.SH")
    T.seq[20, s, 0] = 1.0   # D=19 之后的一天
    T.seq[19, s, 0] = 2.0   # D 当天
    w = deep._windows(T, np.array([19]), np.array([s]))
    assert w[0, -1, 0] == 2.0 and (w[0, :, 0] != 1.0).all()
