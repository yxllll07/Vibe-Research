"""通达信官网每日盘后包:某交易日沪深北全部证券的不复权日线 + 当日简称。

下载的 zip 原样进原始区(带 sha256),解析复用 Vibe-Research 取数层里已经过实测的
`_tdx_parse_package`(残缺包、非有限数值、名称不是 GBK 都会抛错,不把部分市场当全市场)。

单位:个股 volume 为「股」,amount 为「元」(取数层注释与实测一致)。
prev_close 是交易所当日的前收盘(除权除息日为除权参考价),所以
close / prev_close - 1 已是剔除公司行为后的当日收益,涨跌停价也必须以它为基准。
"""
from __future__ import annotations

import sys
import time
from datetime import date, datetime, time as dtime
from pathlib import Path

import pandas as pd
import requests

from ..config import REPO_ROOT, TZ
from . import securities as sec
from .raw_store import RawRecord, RawStore

SOURCE = "tdx_package"
URL = "https://www.tdx.com.cn/products/data/data/g4day/{ymd}.zip"
_SCRIPTS = REPO_ROOT / ".agents" / "skills" / "data-access" / "scripts"


class PackageMissing(LookupError):
    """官网 404:非交易日、尚未发布,或早于官网保留范围。"""


def _parser():
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    from sources.astock_ported import _tdx_parse_package  # type: ignore[import-not-found]

    return _tdx_parse_package


def download(store: RawStore, d: date, *, retries: int = 4, timeout: float = 90) -> RawRecord:
    key = d.strftime("%Y%m%d")
    rec = store.get(SOURCE, key)
    if rec is not None:
        return rec
    url = URL.format(ymd=key)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=(10, timeout))
            if resp.status_code == 404:
                raise PackageMissing(f"{key} 没有通达信盘后包(404)")
            resp.raise_for_status()
            if not resp.content.startswith(b"PK"):
                raise RuntimeError(f"{key} 返回的不是 zip")
            return store.put(SOURCE, key, ".zip", resp.content, url, resp.status_code)
        except PackageMissing:
            raise
        except Exception as exc:  # 网络抖动:指数退避重试
            last = exc
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"{key} 下载失败 {retries} 次: {last}")


def normalize(rec: RawRecord, d: date, available_time: dtime) -> pd.DataFrame:
    rows = _parser()(rec.path.read_bytes(), d.strftime("%Y%m%d"))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"{d} 的盘后包解析为空")
    if (pd.to_datetime(df["date"]).dt.date != d).any():
        raise RuntimeError(f"{d} 的盘后包里混有别的日期")
    df["exchange"] = df["market"].map(sec.EXCHANGE_BY_TDX_MARKET)
    cls = [sec.classify(ex, c) for ex, c in zip(df["exchange"], df["code"])]
    df["security_type"] = [c[0] for c in cls]
    df["board"] = [c[1] for c in cls]
    df = df[df["security_type"].isin(["A", "index"])].copy()
    df["ts_code"] = [sec.ts_code(ex, c) for ex, c in zip(df["exchange"], df["code"])]
    df["is_risk_warning"] = df["name"].map(sec.is_risk_warning)
    df["is_delisting_period"] = df["name"].map(sec.is_delisting_period)
    df["listing_day_marker"] = df["name"].map(sec.is_listing_day_marker)
    df["is_suspended"] = (df["volume"] == 0) & (df["security_type"] == "A")
    df = df.rename(columns={"volume": "volume_shares", "amount": "amount_cny"})
    df["trade_date"] = d
    df["event_time"] = datetime.combine(d, dtime(15, 0), TZ)
    df["available_at"] = datetime.combine(d, available_time, TZ)
    df["ingested_at"] = pd.Timestamp(rec.fetched_at)
    df["source"] = SOURCE
    df["raw_sha256"] = rec.sha256
    cols = ["trade_date", "ts_code", "exchange", "code", "name", "security_type", "board",
            "prev_close", "open", "high", "low", "close", "volume_shares", "amount_cny",
            "is_risk_warning", "is_delisting_period", "listing_day_marker", "is_suspended",
            "event_time", "available_at", "ingested_at", "source", "raw_sha256"]
    if df["ts_code"].duplicated().any():
        raise RuntimeError(f"{d} 的盘后包有重复代码")
    return df[cols].reset_index(drop=True)


def normalized_path(root: Path, d: date) -> Path:
    return root / "normalized" / "daily_bar" / str(d.year) / f"{d:%Y%m%d}.parquet"
