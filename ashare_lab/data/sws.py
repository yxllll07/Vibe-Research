"""申万行业分类(2021 版)个股归属及变迁:swsresearch.com 公开的 StockClassifyUse_stock.xls。

每行 = (股票代码, 计入日期, 行业代码, 更新日期)。
- effective_from = 计入日期;effective_to = 同一股票下一条记录的计入日期。
- first_known_at:没有官方发布时间戳,按「计入当日已可知」处理(申万调整在生效日前后公告)。
  2021 版标准在 2021-07 发布并回溯映射了大量历史记录,早于本项目窗口(2024-07 起),不影响本窗口;
  record_updated_at(更新日期)远晚于计入日期的行在质量报告里单独计数。
- 行业代码 6 位:前 2 位一级、前 4 位二级、6 位三级。

站点证书链缺中间证书:这里随包附带 DigiCert 公开的中间证书
(GeoTrust G2 TLS CN RSA4096 SHA256 2022 CA1,签发者 DigiCert Global Root G2,2032-12-14 到期),
与系统 / 代理 CA 合并后照常校验,不关闭证书校验。
"""
from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from ..config import TZ
from .raw_store import RawStore

SOURCE = "sws_class"
URL = "https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls"
INTERMEDIATE = Path(__file__).parent / "certs" / "GeoTrustG2TLSCNRSA4096SHA2562022CA1.pem"


def _ca_bundle() -> str:
    base = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
    if not base:
        import certifi

        base = certifi.where()
    out = Path(tempfile.gettempdir()) / "ashare_lab_sws_bundle.pem"
    out.write_bytes(Path(base).read_bytes() + b"\n" + INTERMEDIATE.read_bytes())
    return str(out)


def fetch(store: RawStore, retries: int = 3) -> pd.DataFrame:
    key = datetime.now(TZ).strftime("%Y%m%d")
    rec = store.get(SOURCE, key)
    if rec is None:
        last: Exception | None = None
        for _ in range(retries):
            try:
                r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60, verify=_ca_bundle())
                r.raise_for_status()
                if len(r.content) < 100_000:
                    raise RuntimeError(f"申万分类表只有 {len(r.content)} 字节,疑似错误页")
                rec = store.put(SOURCE, key, ".xls", r.content, URL, r.status_code)
                break
            except Exception as exc:
                last = exc
        if rec is None:
            raise RuntimeError(f"申万分类表下载失败: {last}")
    return normalize(pd.read_excel(io.BytesIO(rec.path.read_bytes())), rec.fetched_at)


def _ts_code(code: int | str) -> str:
    c = str(code).zfill(6)
    if c[0] == "6":
        return f"{c}.SH"
    if c[0] in "03":
        return f"{c}.SZ"
    return f"{c}.BJ"


def normalize(raw: pd.DataFrame, fetched_at: str) -> pd.DataFrame:
    df = raw.rename(columns={"股票代码": "code", "计入日期": "start", "行业代码": "industry_code", "更新日期": "updated"})
    need = {"code", "start", "industry_code", "updated"}
    if not need <= set(df.columns):
        raise RuntimeError(f"申万分类表列名变了:{list(raw.columns)}")
    df["ts_code"] = df["code"].map(_ts_code)
    df["start"] = pd.to_datetime(df["start"])
    df = df.sort_values(["ts_code", "start"])
    df["industry_code"] = df["industry_code"].astype(int).astype(str).str.zfill(6)
    out = pd.DataFrame({
        "ts_code": df["ts_code"],
        "sector_id": "SW3_" + df["industry_code"],
        "sw_l1": "SW1_" + df["industry_code"].str[:2],
        "sw_l2": "SW2_" + df["industry_code"].str[:4],
        "provider": "sw2021",
        "effective_from": df["start"].dt.date,
        "effective_to": df.groupby("ts_code")["start"].shift(-1).dt.date,
        "first_known_at": df["start"].dt.tz_localize(TZ),
        "record_updated_at": pd.to_datetime(df["updated"]).dt.tz_localize(TZ),
        "ingested_at": pd.Timestamp(fetched_at),
    })
    return out.reset_index(drop=True)


def daily_l1(members: pd.DataFrame, panel_keys: pd.DataFrame) -> pd.Series:
    """给 (trade_date, ts_code) 面板逐行查当日有效、当时已知的一级行业;查不到为 NA。"""
    m = members.sort_values("effective_from")
    keys = panel_keys[["trade_date", "ts_code"]].copy()
    keys["_d"] = pd.to_datetime(keys["trade_date"])
    m = m.assign(_from=pd.to_datetime(m["effective_from"]), _to=pd.to_datetime(m["effective_to"]))
    merged = pd.merge_asof(keys.reset_index().sort_values("_d"), m.sort_values("_from"),
                           left_on="_d", right_on="_from", by="ts_code", direction="backward")
    valid = merged["_to"].isna() | (merged["_d"] < merged["_to"])
    # 以 D 日收盘后决策:first_known_at 须不晚于 D 日 21:00
    known = merged["first_known_at"] <= (merged["_d"] + pd.Timedelta(hours=21)).dt.tz_localize(TZ)
    merged["sw_l1"] = merged["sw_l1"].where(valid & known)
    return merged.set_index("index")["sw_l1"].reindex(panel_keys.index)
