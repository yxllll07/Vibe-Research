"""全市场广度与连板梯队(观测量,不是主观的情绪标签)。

统计口径:当日有成交的沪深主板非风险警示 A 股。每个比例都写明分母;分母为 0 时为 NA,不是 0。
「昨日首板晋级率」= 昨日收盘首板(连板数=1)今日继续收盘涨停的比例,分母是昨日首板家数。
"""
from __future__ import annotations

import pandas as pd


def market_features(ev: pd.DataFrame) -> pd.DataFrame:
    base = ev[(ev["board"] == "main") & (ev["security_type"] == "A") & ~ev["is_risk_warning"]
              & ~ev["is_delisting_period"] & ~ev["is_suspended"]].copy()
    base = base.sort_values(["ts_code", "trade_date"])
    base["prev_streak"] = base.groupby("ts_code")["limit_streak"].shift(1).fillna(0).astype(int)
    rows = []
    for d, day in base.groupby("trade_date", sort=True):
        n = len(day)
        lu = int(day["limit_up_close"].sum())
        broke = int(day["broke_board"].sum())
        touched = lu + broke
        streak = day["limit_streak"]
        y1 = day[day["prev_streak"] == 1]
        y2 = day[day["prev_streak"] == 2]
        rows.append({
            "trade_date": d,
            "mkt_n_traded": n,
            "mkt_up_ratio": (day["close"] > day["prev_close"]).mean() if n else pd.NA,
            "mkt_limit_up": lu,
            "mkt_limit_down": int(day["limit_down_close"].sum()),
            "mkt_broke_rate": broke / touched if touched else pd.NA,
            "mkt_board_1": int((streak == 1).sum()),
            "mkt_board_2": int((streak == 2).sum()),
            "mkt_board_3": int((streak == 3).sum()),
            "mkt_board_4p": int((streak >= 4).sum()),
            "mkt_max_height": int(streak.max()) if n else 0,
            "mkt_promo_1to2": y1["limit_up_close"].mean() if len(y1) else pd.NA,
            "mkt_promo_2to3": y2["limit_up_close"].mean() if len(y2) else pd.NA,
            "mkt_amount_cny": float(day["amount_cny"].sum()),
            "mkt_ret_median": float((day["close"] / day["prev_close"] - 1).median()),
            "mkt_ret_dispersion": float((day["close"] / day["prev_close"] - 1).std()),
        })
    out = pd.DataFrame(rows)
    out["mkt_amount_chg_5"] = out["mkt_amount_cny"] / out["mkt_amount_cny"].rolling(5, min_periods=5).mean() - 1
    return out


MARKET_FEATURES = [
    "mkt_up_ratio", "mkt_limit_up", "mkt_limit_down", "mkt_broke_rate", "mkt_board_1", "mkt_board_2", "mkt_board_3",
    "mkt_board_4p", "mkt_max_height", "mkt_promo_1to2", "mkt_promo_2to3", "mkt_ret_median", "mkt_ret_dispersion",
    "mkt_amount_chg_5",
]
