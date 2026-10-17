"""D 日盘后冻结票池与 T 日 9:25 二阶段过滤。

- 冻结文件只增不改:frozen.v1.json、frozen.v2.json……新版本记录上一版的 sha256。
- 冻结时刻不得晚于 T 日 9:14:59。
- 二阶段过滤只能在冻结集合内通过 / 放弃,集合外的代码一律拒绝;没有竞价数据输出 NO_AUCTION_DATA,
  不用 T 日 OHLCV 倒填。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Any

from ..config import TZ
from ..data.auction import AuctionFinal

FREEZE_DEADLINE = dtime(9, 14, 59)
MAX_PER_MODE = 2
MODES = ("A", "B", "C", "D", "E")


class FreezeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    mode: str
    rank: int
    ts_code: str
    score: float
    evidence_features: dict[str, Any]
    eligible: bool = True
    forbidden_reason: str | None = None
    data_quality_flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FrozenPool:
    model_version: str
    data_snapshot_id: str
    signal_date: date
    target_date: date
    signal_asof: datetime
    freeze_at: datetime
    price_data_cutoff: date
    candidates: tuple[Candidate, ...]

    def validate(self) -> None:
        if self.target_date <= self.signal_date:
            raise FreezeError("目标交易日必须晚于信号日")
        deadline = datetime.combine(self.target_date, FREEZE_DEADLINE, TZ)
        if self.freeze_at > deadline:
            raise FreezeError(f"冻结时刻 {self.freeze_at} 晚于 {deadline}")
        if self.price_data_cutoff > self.signal_date:
            raise FreezeError("价格数据截止日晚于信号日")
        per_mode: dict[str, int] = {}
        for c in self.candidates:
            if c.mode not in MODES:
                raise FreezeError(f"未知模式 {c.mode}")
            per_mode[c.mode] = per_mode.get(c.mode, 0) + 1
            if per_mode[c.mode] > MAX_PER_MODE:
                raise FreezeError(f"模式 {c.mode} 候选超过 {MAX_PER_MODE} 只")

    def codes(self) -> set[str]:
        return {c.ts_code for c in self.candidates}

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["candidates"] = [asdict(c) for c in self.candidates]
        return json.loads(json.dumps(d, default=str))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_frozen(pool: FrozenPool, pools_root: Path) -> Path:
    pool.validate()
    day = pools_root / f"{pool.target_date:%Y%m%d}"
    day.mkdir(parents=True, exist_ok=True)
    versions = sorted(day.glob("frozen.v*.json"), key=lambda p: int(p.stem.split(".v")[1]))
    n = len(versions) + 1
    out = day / f"frozen.v{n}.json"
    if out.exists():
        raise FreezeError(f"{out} 已存在,冻结文件不可覆盖")
    payload = pool.to_json()
    payload["version"] = n
    payload["previous_version_sha256"] = _sha(versions[-1]) if versions else None
    with out.open("x", encoding="utf-8") as fh:  # "x":文件存在就失败
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return out


@dataclass(frozen=True)
class AuctionDecision:
    ts_code: str
    observed_at: datetime | None
    auction_snapshot_at: datetime | None
    auction_price: float | None
    gap_pct: float | None
    auction_amount: float | None
    auction_volume_ratio: float | None
    model_preopen_score: float
    execution_allowed: bool
    rejection_reason: str | None
    recommended_order_start_at: datetime | None


def auction_filter(pool: FrozenPool, auctions: dict[str, AuctionFinal], prev_close: dict[str, float],
                   requested: list[str] | None = None, max_gap_pct: float = 9.5) -> list[AuctionDecision]:
    """只对冻结候选逐只判通过 / 放弃;requested 里出现冻结集合外的代码直接报错。"""
    frozen = {c.ts_code: c for c in pool.candidates}
    codes = requested if requested is not None else list(frozen)
    outside = [c for c in codes if c not in frozen]
    if outside:
        raise FreezeError(f"{outside} 不在冻结票池里,盘前不得临时新增")
    out = []
    for code in codes:
        cand = frozen[code]
        a = auctions.get(code)
        if a is None:
            out.append(AuctionDecision(code, None, None, None, None, None, None, cand.score, False,
                                       "NO_AUCTION_DATA", None))
            continue
        gap = (a.price / prev_close[code] - 1) * 100
        allowed = cand.eligible and gap < max_gap_pct
        reason = None if allowed else (cand.forbidden_reason or f"GAP_{gap:.2f}PCT_TOO_HIGH")
        out.append(AuctionDecision(code, a.observed_available_at, a.underlying_event_at, a.price, round(gap, 3),
                                   a.amount_cny, a.volume_ratio, cand.score, allowed, reason,
                                   a.observed_available_at if allowed else None))
    return out
