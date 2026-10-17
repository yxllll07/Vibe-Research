"""把归一化的逐日 parquet 读成面板(DuckDB 直接扫文件,不必整库入内存)。"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd


def load_panel(root: Path, start: date, end: date, *, main_board_only: bool = False,
               columns: list[str] | None = None) -> pd.DataFrame:
    glob = str(root / "normalized" / "daily_bar" / "*" / "*.parquet")
    cols = ", ".join(columns) if columns else "*"
    where = "trade_date BETWEEN ? AND ? AND security_type = 'A'"
    if main_board_only:
        where += " AND board = 'main'"
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT {cols} FROM read_parquet('{glob}') WHERE {where}", [start, end]).df()
    finally:
        con.close()
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    # 通达信盘后包里停牌股的开 / 高 / 低记为 0(收盘 = 前收):这是「无成交」不是价格 0,按缺失处理
    if "is_suspended" in df.columns:
        for col in ("open", "high", "low"):
            if col in df.columns:
                df.loc[df["is_suspended"] & (df[col] <= 0), col] = float("nan")
    return df
