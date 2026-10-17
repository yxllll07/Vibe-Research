"""一键构建研究面板:日线 → 涨跌停事件 → 个股 / 行业 / 市场特征;标签单独成表。

输出(<数据根>/features/<experiment>/):
- features.parquet  键 (trade_date, ts_code),只含 D 日盘后可见的信息 + eligible / forbidden_reason
- labels.parquet    键 (signal_date, ts_code),只含 T 日及以后的结果
- manifest.json     输入文件 sha256 汇总、覆盖范围、配置哈希、行数
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from .config import Config
from .data.ingest import sectors_path
from .data.panel import load_panel
from .data.sws import daily_l1
from .features.events import add_limit_events
from .features.market import market_features
from .features.sector import add_sector_features
from .features.stock import add_stock_features
from .labels import build_labels
from .universe.eligibility import build_security_master, eligible_mask, listed_trading_days

log = logging.getLogger(__name__)


def features_dir(cfg: Config) -> Path:
    return cfg.data_root / "features" / cfg["experiment_id"]


def build(cfg: Config, start: date, end: date) -> Path:
    root = cfg.data_root
    log.info("读日线面板 %s ~ %s", start, end)
    panel = load_panel(root, start, end, main_board_only=True)
    if panel.empty:
        raise RuntimeError("面板为空,先运行 ingest")
    days = sorted(panel["trade_date"].unique())
    master = build_security_master(panel, days[0], days[-1])
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    ld = listed_trading_days(panel, master)
    panel["listed_trading_days"] = ld
    log.info("涨跌停事件 %d 行", len(panel))
    ev = add_limit_events(panel, ld)
    log.info("个股特征")
    feats = add_stock_features(ev)
    log.info("行业归属与行业特征")
    members = pd.read_parquet(sectors_path(root))
    feats["sw_l1"] = daily_l1(members, feats).values
    feats = add_sector_features(feats)
    log.info("市场特征")
    mkt = market_features(ev)
    feats = feats.merge(mkt, on="trade_date", how="left")
    feats["eligible"] = eligible_mask(feats, feats["listed_trading_days"])
    labels = build_labels(ev, tuple(cfg["features"]["horizons_trading_days"]))

    out = features_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    drop = [c for c in ("event_time", "ingested_at", "source", "raw_sha256", "log_ret_1", "adj_index") if c in feats]
    feats.drop(columns=drop).to_parquet(out / "features.parquet", index=False)
    labels.to_parquet(out / "labels.parquet", index=False)
    master.to_parquet(out / "security_master.parquet", index=False)
    shas = sorted(panel["raw_sha256"].unique())
    manifest = {
        "experiment_id": cfg["experiment_id"], "config_hash": cfg.hash(),
        "range": [str(days[0]), str(days[-1])], "trading_days": len(days),
        "feature_rows": int(len(feats)), "eligible_rows": int(feats["eligible"].sum()), "label_rows": int(len(labels)),
        "raw_inputs": len(shas), "raw_inputs_sha256_digest": hashlib.sha256("".join(shas).encode()).hexdigest(),
        "sw_membership_rows": int(len(members)),
        "disabled_feature_groups": cfg["features"].get("disabled_feature_groups", []),
        "unknown_fields": ["open_times", "first_limit_time"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
