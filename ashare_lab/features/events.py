"""涨跌停事件(由不复权价 + 当日前收 + 按日期生效的涨跌幅规则重建)。

日线只能判断收盘封板与盘中触板,不知道开板次数和首封时间:这两列一律 UNKNOWN(NA),不填 0。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..rules.limits import band_prices, limit_pct


def add_limit_events(panel: pd.DataFrame, listed_days: pd.Series | None = None) -> pd.DataFrame:
    df = panel.copy()
    ld = listed_days if listed_days is not None else pd.Series(pd.NA, index=df.index)
    pcts = []
    for board, st, d, n in zip(df["board"], df["is_risk_warning"], df["trade_date"], ld):
        if board not in ("main", "chinext", "star", "bse"):
            pcts.append(np.nan)
            continue
        p = limit_pct(board, bool(st), d, None if pd.isna(n) else int(n))
        pcts.append(np.nan if p is None else p)
    df["limit_pct"] = pcts
    # 涨跌停价只取决于 (前收, 比例):对去重后的组合用 Decimal 精确取整,再映射回每行
    cache: dict[tuple[float, float], tuple[float, float]] = {}
    ups, downs = [], []
    for pc, p in zip(df["prev_close"], df["limit_pct"]):
        if np.isnan(p) or not pc or pc <= 0:
            ups.append(np.nan); downs.append(np.nan)
            continue
        key = (pc, p)
        if key not in cache:
            cache[key] = band_prices(pc, p)
        u, dn = cache[key]
        ups.append(u); downs.append(dn)
    df["up_limit"] = ups
    df["down_limit"] = downs
    traded = ~df["is_suspended"]
    eq = lambda a, b: (a - b).abs() < 1e-6  # noqa: E731
    df["limit_up_close"] = traded & eq(df["close"], df["up_limit"])
    df["limit_down_close"] = traded & eq(df["close"], df["down_limit"])
    df["touched_up"] = traded & eq(df["high"], df["up_limit"])
    df["broke_board"] = df["touched_up"] & ~df["limit_up_close"]
    df["one_word_up"] = traded & eq(df["open"], df["up_limit"]) & eq(df["low"], df["up_limit"])
    df["one_word_down"] = traded & eq(df["open"], df["down_limit"]) & eq(df["high"], df["down_limit"])
    df["open_times"] = pd.array([pd.NA] * len(df), dtype="Int64")      # UNKNOWN:日线不可得
    df["first_limit_time"] = pd.array([pd.NA] * len(df), dtype="string")  # UNKNOWN:日线不可得
    df = df.sort_values(["ts_code", "trade_date"])
    # 连板数:按该股自己的交易序列连续收盘涨停计数
    grp = (~df["limit_up_close"]).groupby(df["ts_code"]).cumsum()
    df["limit_streak"] = df.groupby([df["ts_code"], grp])["limit_up_close"].cumsum().where(df["limit_up_close"], 0).astype(int)
    return df
