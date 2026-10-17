"""个股截至 D 日收盘的特征(只用 trade_date <= D 的行;滚动窗口全部向后看)。

收益用 close / prev_close 逐日连乘(prev_close 为交易所前收,除权日已调整),
不使用今天下载的前复权序列。换手率 / 市值类特征在 v0.1 被配置关闭(无历史股本数据)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_stock_features(ev: pd.DataFrame) -> pd.DataFrame:
    df = ev.sort_values(["ts_code", "trade_date"]).copy()
    g = df.groupby("ts_code", sort=False)
    df["ret_1"] = df["close"] / df["prev_close"] - 1
    df["log_ret_1"] = np.log1p(df["ret_1"])
    df["adj_index"] = g["log_ret_1"].cumsum().pipe(np.exp)  # 只在同一只股票内部比较
    gi = df.groupby("ts_code", sort=False)["adj_index"]
    for n in (3, 5, 10, 20, 60):
        df[f"ret_{n}"] = df["adj_index"] / gi.shift(n) - 1
    df["amplitude"] = (df["high"] - df["low"]) / df["prev_close"]
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["body"] = (df["close"] - df["open"]) / df["prev_close"]
    df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["prev_close"]
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["prev_close"]
    df["close_pos_in_range"] = ((df["close"] - df["low"]) / rng).fillna(0.5)
    df["log_amount"] = np.log1p(df["amount_cny"])
    ga = df.groupby("ts_code", sort=False)["amount_cny"]
    for n in (5, 20):
        mean = ga.transform(lambda s: s.rolling(n, min_periods=n).mean())
        df[f"amount_ratio_{n}"] = df["amount_cny"] / mean
    for n in (5, 20, 60):
        ma = gi.transform(lambda s: s.rolling(n, min_periods=n).mean())
        df[f"close_to_ma_{n}"] = df["adj_index"] / ma - 1
    peak = gi.transform(lambda s: s.rolling(20, min_periods=20).max())
    df["drawdown_20"] = df["adj_index"] / peak - 1
    gl = df.groupby("ts_code", sort=False)
    df["limit_up_count_20"] = gl["limit_up_close"].transform(lambda s: s.rolling(20, min_periods=1).sum())
    df["broke_count_5"] = gl["broke_board"].transform(lambda s: s.rolling(5, min_periods=1).sum())
    df["prev_limit_up"] = gl["limit_up_close"].shift(1).fillna(False).astype(bool)
    df["prev_streak"] = gl["limit_streak"].shift(1).fillna(0).astype(int)
    df["history_days"] = gl.cumcount() + 1
    # 截面分位(同一交易日内),只用当日已发生的数据
    for col in ("ret_5", "ret_20", "log_amount", "amount_ratio_5"):
        df[f"{col}_xs_rank"] = df.groupby("trade_date")[col].rank(pct=True)
    return df


STOCK_FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60", "amplitude", "body", "upper_shadow", "lower_shadow",
    "close_pos_in_range", "log_amount", "amount_ratio_5", "amount_ratio_20", "close_to_ma_5", "close_to_ma_20",
    "close_to_ma_60", "drawdown_20", "limit_up_count_20", "broke_count_5", "limit_streak", "prev_streak",
    "limit_up_close", "broke_board", "touched_up", "one_word_up", "limit_down_close",
    "ret_5_xs_rank", "ret_20_xs_rank", "log_amount_xs_rank", "amount_ratio_5_xs_rank",
]
