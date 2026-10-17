"""按交易日逐日拉取并归一化;可断点续传(已落盘的原始包直接复用)。"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime
from pathlib import Path

import pandas as pd

from ..config import TZ, Config
from . import calendar as cal
from . import sws, tdx
from .raw_store import RawStore

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    start: str
    end: str
    trading_days: int = 0
    ok: list[str] = field(default_factory=list)
    missing_404: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "start": self.start, "end": self.end, "trading_days": self.trading_days,
            "ok": len(self.ok), "missing_404": self.missing_404, "failed": self.failed,
            "coverage": round(len(self.ok) / self.trading_days, 4) if self.trading_days else None,
        }


def calendar_path(root: Path) -> Path:
    return root / "normalized" / "calendar.parquet"


def ingest_calendar(cfg: Config) -> pd.DataFrame:
    root = cfg.data_root
    df = cal.fetch_calendar(RawStore(root))
    out = calendar_path(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def sectors_path(root: Path) -> Path:
    return root / "normalized" / "sw_membership.parquet"


def ingest_sectors(cfg: Config) -> pd.DataFrame:
    root = cfg.data_root
    df = sws.fetch(RawStore(root))
    out = sectors_path(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def ingest_tdx(cfg: Config, start: date, end: date, *, force_normalize: bool = False) -> IngestReport:
    root = cfg.data_root
    store = RawStore(root)
    cpath = calendar_path(root)
    calendar = cal.TradingCalendar.load(cpath) if cpath.exists() else cal.TradingCalendar.from_frame(ingest_calendar(cfg))
    avail = dtime.fromisoformat(cfg["data"]["tdx_available_at_local_time"])
    days = calendar.between(start, end)
    today = datetime.now(TZ).date()
    days = [d for d in days if d < today]  # 当天的包收盘后数小时才发布,不抓未来
    report = IngestReport(str(start), str(end), len(days))
    for i, d in enumerate(days, 1):
        key = d.strftime("%Y%m%d")
        out = tdx.normalized_path(root, d)
        try:
            rec = tdx.download(store, d)
            if force_normalize or not out.exists():
                df = tdx.normalize(rec, d, avail)
                out.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out, index=False)
            report.ok.append(key)
        except tdx.PackageMissing:
            report.missing_404.append(key)
        except Exception as exc:
            report.failed[key] = f"{type(exc).__name__}: {exc}"[:300]
        if i % 20 == 0 or i == len(days):
            log.info("tdx %d/%d ok=%d 404=%d failed=%d", i, len(days), len(report.ok),
                     len(report.missing_404), len(report.failed))
    rep_path = root / "reports" / f"ingest_tdx_{start:%Y%m%d}_{end:%Y%m%d}.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(report.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    return report
