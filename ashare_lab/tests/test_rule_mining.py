from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from ashare_lab.evaluation.rule_mining import CANDIDATE_FEATURES, mine_rules, rule_scores


def _synthetic(n_days=80, n=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    d0 = date(2025, 1, 1)
    for i in range(n_days):
        d = d0 + timedelta(days=i)
        for j in range(n):
            row = {f: rng.normal() for f in CANDIDATE_FEATURES}
            row.update(trade_date=d, ts_code=f"{600000 + j}.SH", mode="B")
            # 只有 amount_ratio_5 真正预测收益(负向),其余为噪声
            row["r_open_next_open"] = -0.02 * row["amount_ratio_5"] + rng.normal(scale=0.02)
            rows.append(row)
    return pd.DataFrame(rows)


def test_mines_true_signal_with_correct_sign():
    df = _synthetic()
    spec = mine_rules(df, df.trade_date.min(), df.trade_date.max(), "h")
    chosen = spec["modes"]["B"]["chosen"]
    assert chosen and chosen[0]["feature"] == "amount_ratio_5" and chosen[0]["sign"] == -1
    assert all(abs(c["t"]) >= 2 for c in chosen)
    assert spec["modes"]["A"]["chosen"] == []  # 没有样本的模式不出规则 → 不出票
    s = rule_scores(df, spec)["B"]
    assert s.notna().all() and s.between(0, 1).all()


def test_refuses_rows_outside_training_period():
    df = _synthetic(n_days=10)
    with pytest.raises(ValueError):
        mine_rules(df, df.trade_date.min(), df.trade_date.max() - timedelta(days=1), "h")
