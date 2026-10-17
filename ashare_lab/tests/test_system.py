"""交易系统 v1:策略定义不可篡改、状态模型 JSON 往返一致。"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from ashare_lab.evaluation.regime import fit_model
from ashare_lab.features.market import MARKET_FEATURES
from ashare_lab.system import strategy as st


def _market(n=60, seed=1):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-01-01", periods=n, freq="B").date
    return pd.DataFrame({"trade_date": days, **{c: rng.normal(size=n) for c in MARKET_FEATURES}})


def test_regime_json_roundtrip_predicts_identically():
    m = _market()
    model = fit_model(m, m.trade_date.iloc[39])
    back = st._gmm_from_json(json.loads(json.dumps(st._gmm_to_json(model))))
    assert (model.predict(m).values == back.predict(m).values).all()


def test_strategy_tamper_detected(tmp_path, monkeypatch):
    m = _market()
    model = fit_model(m, m.trade_date.iloc[39])
    body = {"version": "strategy_v1", "rules": {"modes": {}}, "regime_model": st._gmm_to_json(model),
            "tradeable_states": {k: [0] for k in "ABCDE"}, "params": {"regime_gate": True}}
    spec = {**body, "hash": st._hash(body)}
    d = tmp_path / "system" / "strategy_v1"
    d.mkdir(parents=True)
    (d / "strategy.json").write_text(json.dumps(spec, default=str))
    monkeypatch.setenv("ASHARE_LAB_DATA", str(tmp_path))
    from ashare_lab.config import load_config
    cfg = load_config()
    s = st.Strategy.load(cfg)
    assert s.tradeable("A", 0) and not s.tradeable("A", 1)
    spec["tradeable_states"]["A"] = [0, 1, 2, 3]  # 事后放宽却不重算哈希
    (d / "strategy.json").write_text(json.dumps(spec, default=str))
    with pytest.raises(RuntimeError):
        st.Strategy.load(cfg)


def test_gate_off_trades_every_state():
    spec = {"params": {"regime_gate": False}, "tradeable_states": {"B": [2]}}
    s = st.Strategy(spec, None)  # type: ignore[arg-type]
    assert s.tradeable("B", 0) and not s.gate_says("B", 0) and s.gate_says("B", 2)
