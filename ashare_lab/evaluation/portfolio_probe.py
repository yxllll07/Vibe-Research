"""探索:放宽「每模式每天 1 只」,在信号稳定的池(B、E)里等权持有分数前 N 只。

⚠️ 这是看过 eod_v1 盲测结果之后提出的(事后假设)。只以验证期作判断依据;
2026-04 ~ 10 的数字标为「已揭盲期,非合法盲测」,真正的检验要靠 2026-10-19 以后的前向数据。
N 事先固定为 5 和 10,两档都报告,不挑。

口径:信号日 D 选股,T 日开盘买入;B 持有到 T+1 开盘,E 持有 5 个交易日(T+4 收盘)。
T 日一字涨停 / 停牌视为买不到,该仓位收益记 0(资金闲置)。每笔扣往返成本 0.15%。
同时给出:池内全部等权(基准)与分数最低 N 只(对照)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import ROUND_TRIP_COST, TARGETS
from .stats import block_bootstrap_ci

NS = (5, 10)
MODES = ("B", "E")


def _net(df: pd.DataFrame, mode: str) -> pd.Series:
    r = df[TARGETS[mode]] - ROUND_TRIP_COST
    unfillable = df["t_one_word_up"].astype("boolean").fillna(False) | df["t_suspended"].astype("boolean").fillna(False)
    return r.where(~unfillable, 0.0)


def probe(data: pd.DataFrame, score_col: str, split: str) -> pd.DataFrame:
    rows = []
    sub = data[(data["split"] == split)]
    for mode in MODES:
        m = sub[(sub["mode"] == mode) & sub[TARGETS[mode]].notna()].dropna(subset=[score_col]).copy()
        m["net"] = _net(m, mode)
        g = m.sort_values(["trade_date", score_col], ascending=[True, False]).groupby("trade_date")
        series = {"pool_all": g["net"].mean()}
        for n in NS:
            series[f"top{n}"] = g.head(n).groupby("trade_date")["net"].mean()
            series[f"bottom{n}"] = m.sort_values(["trade_date", score_col]).groupby("trade_date").head(n).groupby("trade_date")["net"].mean()
        hold = 1 if TARGETS[mode] == "r_open_next_open" else 5
        for name, s in series.items():
            ci = block_bootstrap_ci(s, block=max(5, hold))
            rows.append({"split": split, "mode": mode, "portfolio": name, "days": int(len(s)),
                         "mean_net_per_trade": float(s.mean()), "ci": ci.get("ci"),
                         "p_mean_le_0": ci.get("p_mean_le_0"),
                         "hit_rate_days": float((s > 0).mean()),
                         # 不重叠近似:每 hold 天换一次仓的累计收益
                         "cum_nonoverlap": float(np.prod(1 + s.iloc[::hold].to_numpy()) - 1)})
    return pd.DataFrame(rows)
