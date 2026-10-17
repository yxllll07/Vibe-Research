"""按交易日分块的自助法:日收益均值与配置间差异的置信区间(小样本、序列相关下比 iid 更保守)。"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20261017


def _blocks(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, max(n - block + 1, 1), size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]


def block_bootstrap_ci(daily: pd.Series, block: int = 5, reps: int = 2000, alpha: float = 0.05) -> dict:
    x = daily.dropna().to_numpy()
    if len(x) < 2 * block:
        return {"n_days": int(len(x)), "mean": float(np.mean(x)) if len(x) else None, "ci": None}
    rng = np.random.default_rng(SEED)
    means = np.array([x[_blocks(len(x), block, rng)].mean() for _ in range(reps)])
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"n_days": int(len(x)), "mean": float(x.mean()), "ci": [float(lo), float(hi)],
            "p_mean_le_0": float((means <= 0).mean())}


def daily_returns(equity: pd.DataFrame, capital: float) -> pd.Series:
    e = equity.set_index("trade_date")["equity"]
    return e.pct_change().fillna(e.iloc[0] / capital - 1)
