"""T 日 9:25 最终集合竞价快照的 schema 与校验(第一轮只建接口,不启用数据源)。

只接受 9:25 最终撮合结果。任何 9:20→9:25 的路径量(G20 / G24 / 竞价过程价 / ΔGap 等)
都不在白名单里,出现即拒收;字段白名单之外的列一律拒收,而不是忽略。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as dtime

UNDERLYING_EVENT = dtime(9, 25)
EARLIEST_OBSERVED = dtime(9, 26)

ALLOWED_FIELDS = frozenset({
    "trade_date", "ts_code", "price", "vol_shares", "amount_cny", "turnover_rate", "volume_ratio",
    "source", "observed_available_at", "underlying_event_at",
})
REQUIRED_FIELDS = frozenset({"trade_date", "ts_code", "price", "source", "observed_available_at", "underlying_event_at"})


class AuctionSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class AuctionFinal:
    trade_date: str
    ts_code: str
    price: float
    source: str
    observed_available_at: datetime
    underlying_event_at: datetime
    vol_shares: int | None = None
    amount_cny: float | None = None
    turnover_rate: float | None = None
    volume_ratio: float | None = None


def validate_auction_record(rec: dict) -> AuctionFinal:
    extra = set(rec) - ALLOWED_FIELDS
    if extra:
        raise AuctionSchemaError(f"竞价记录含白名单外字段 {sorted(extra)}(只收 9:25 最终撮合,不收竞价路径)")
    missing = REQUIRED_FIELDS - set(rec)
    if missing:
        raise AuctionSchemaError(f"竞价记录缺字段 {sorted(missing)}")
    ev: datetime = rec["underlying_event_at"]
    obs: datetime = rec["observed_available_at"]
    if ev.time().replace(microsecond=0) != UNDERLYING_EVENT:
        raise AuctionSchemaError(f"underlying_event_at={ev} 不是 9:25 最终撮合")
    if obs.time() < EARLIEST_OBSERVED or obs.date() != ev.date():
        raise AuctionSchemaError(f"observed_available_at={obs} 早于 9:26 或不在同一交易日")
    if rec["price"] is None or rec["price"] <= 0:
        raise AuctionSchemaError("竞价价格无效")
    return AuctionFinal(**rec)
