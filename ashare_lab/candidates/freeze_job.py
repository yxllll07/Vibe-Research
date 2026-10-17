"""用某次实验训练好的模型,为信号日 D 生成冻结票池 frozen.vN.json(五模式各最多 2 只 + 槽位选择)。

- 实时用法:D 日盘后(数据已入库)运行,freeze_at = 当前时刻,必须早于 T 日 9:14:59。
- 历史复现(--replay):freeze_at 记为 D 日 21:00 并在 data_quality_flags 里标 HISTORICAL_REPLAY。
"""
from __future__ import annotations

import json
import pickle
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..build import features_dir
from ..config import TZ, Config
from ..data.calendar import TradingCalendar
from ..data.ingest import calendar_path
from ..data.pit import d_postclose
from .freeze import Candidate, FrozenPool, write_frozen
from .modes import candidate_rows, pick_slots, top_k

EVIDENCE_COLS = ["close", "ret_1", "ret_5", "limit_streak", "prev_streak", "amount_ratio_5", "close_pos_in_range",
                 "sw_l1", "sec_strength_rank", "sec_limit_up", "mkt_limit_up", "mkt_broke_rate"]


def _plain(v: Any) -> Any:
    """numpy / pandas 标量 → JSON 可写的 Python 值;缺失 → None。"""
    if pd.isna(v):
        return None
    return v.item() if hasattr(v, "item") else v


def latest_run(cfg: Config) -> Path:
    runs = sorted((cfg.data_root / "experiments" / cfg["experiment_id"]).glob("*/models.pkl"))
    if not runs:
        raise RuntimeError("没有训练好的实验,先运行 experiment")
    return runs[-1].parent


def freeze(cfg: Config, signal_date: date, score_config: str = "m1_gbm_full", run_dir: Path | None = None,
           replay: bool = False) -> Path:
    run_dir = run_dir or latest_run(cfg)
    feats = pd.read_parquet(features_dir(cfg) / "features.parquet")
    day = feats[feats["trade_date"] == signal_date]
    if day.empty:
        raise RuntimeError(f"特征表里没有 {signal_date}:先 ingest + build")
    cal = TradingCalendar.load(calendar_path(cfg.data_root))
    target = cal.next(signal_date)
    rules_path = run_dir / "rules.json"
    rules = json.loads(rules_path.read_text(encoding="utf-8")) if rules_path.exists() else None
    cands = candidate_rows(day, rules)
    flags = [f"RULE_{cands['rule_version'].iloc[0] if len(cands) else 'NA'}", "OPEN_TIMES_UNKNOWN",
             "NO_TURNOVER_MARKETCAP"]
    if score_config == "m0":
        cands["score"] = cands["m0_score"]
        model_version = f"m0@{run_dir.name}"
    else:
        kind, fs = score_config.split("_")[1:]
        with (run_dir / "models.pkl").open("rb") as fh:
            models = pickle.load(fh)
        rows = cands.merge(day, on=["trade_date", "ts_code"], how="left")
        cands["score"] = float("nan")
        for m in models:
            if m.kind == kind and m.feature_set == fs:
                idx = rows["mode"] == m.mode
                cands.loc[idx.values, "score"] = m.predict(rows[idx])
        model_version = f"{score_config}@{run_dir.name}"
    pool = top_k(cands[["trade_date", "ts_code", "mode", "score"]], "score", k=cfg["pool"]["max_candidates_per_mode"])
    slots = pick_slots(pool, "score").set_index(["mode", "ts_code"])["slot"].to_dict()
    ev = day.set_index("ts_code")
    candidates = tuple(
        Candidate(mode=r.mode, rank=int(r.rank), ts_code=r.ts_code, score=float(r.score),
                  evidence_features={c: _plain(ev.at[r.ts_code, c]) for c in EVIDENCE_COLS},
                  eligible=bool(ev.at[r.ts_code, "eligible"]),
                  data_quality_flags=flags + ([f"SLOT_{slots[(r.mode, r.ts_code)].upper()}"]
                                              if (r.mode, r.ts_code) in slots else []))
        for r in pool.itertuples(index=False))
    freeze_at = d_postclose(signal_date) if replay else datetime.now(TZ)
    if replay:
        candidates = tuple(Candidate(**{**c.__dict__, "data_quality_flags": c.data_quality_flags + ["HISTORICAL_REPLAY"]})
                           for c in candidates)
    fp = FrozenPool(model_version=model_version, data_snapshot_id=f"{cfg['experiment_id']}:{cfg.hash()}",
                    signal_date=signal_date, target_date=target, signal_asof=d_postclose(signal_date),
                    freeze_at=freeze_at, price_data_cutoff=signal_date, candidates=candidates)
    return write_frozen(fp, cfg.data_root / "artifacts" / "pools")
