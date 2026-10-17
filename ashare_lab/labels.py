"""标签 = 未来结果,只放在独立的标签表里,键为 (signal_date=D, ts_code)。

特征表与标签表只能按 (signal_date, ts_code) 对齐;标签里的任何列都不得回流到 D 日特征。
- y_A:D 日未收盘涨停,T 日收盘 == T 日涨停价。
- y_B:D 日收盘首板(连板数=1),T 日继续收盘涨停。
- y_2to3:D 日二连板,T 日继续收盘涨停(独立实验,不混入 B)。
- r_h{n}:T 日开盘买入(代理价)持有到 T+n-1 日收盘的收益,用逐日前收连乘剔除公司行为。
- t_one_word_up:T 日一字涨停 → 代理回测里买不到,收益不能计入。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_labels(ev: pd.DataFrame, horizons: tuple[int, ...] = (1, 3, 5)) -> pd.DataFrame:
    df = ev.sort_values(["ts_code", "trade_date"]).copy()
    g = df.groupby("ts_code", sort=False)
    df["_logr"] = np.log(df["close"] / df["prev_close"])
    df["_cum"] = g["_logr"].cumsum()
    nxt = lambda col, k=1: g[col].shift(-k)  # noqa: E731
    out = pd.DataFrame({"signal_date": df["trade_date"], "ts_code": df["ts_code"]})
    out["target_date"] = nxt("trade_date")
    t_lu = nxt("limit_up_close")
    out["y_A"] = t_lu.where(~df["limit_up_close"])  # D 日已涨停的样本不属于 A,置 NA
    out["y_B"] = t_lu.where(df["limit_streak"] == 1)
    out["y_2to3"] = t_lu.where(df["limit_streak"] == 2)
    out["t_one_word_up"] = nxt("one_word_up")
    out["t_suspended"] = nxt("is_suspended")
    # T 日开盘在调整后价格坐标里的位置:cum_{D} + log(open_T / prev_close_T)
    open_log_t = df["_cum"] + np.log(nxt("open") / nxt("prev_close"))
    for h in horizons:
        out[f"r_h{h}"] = np.exp(nxt("_cum", h) - open_log_t) - 1
    return out.dropna(subset=["target_date"]).reset_index(drop=True)
