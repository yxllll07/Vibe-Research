"""按成交日期分段生效的费用表。

数值是公开口径的默认值,佣金因券商而异,一律可在配置里覆盖;回测报告必须附上实际使用的费率表。
- 印花税(卖出单边):2023-08-28 起 0.05%,之前 0.1%。
- 过户费(双边):2022-04-29 起 0.001%,之前 0.002%。
- 交易所经手费(双边):2023-08-28 起 0.00341%,之前 0.00487%。
- 佣金:默认万 2.5、单笔最低 5 元(券商协议价,非法定)。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Any


@dataclass(frozen=True)
class FeeSchedule:
    effective_from: date
    commission_rate: float = 0.00025
    min_commission_cny: float = 5.0
    stamp_duty_sell: float = 0.0005
    transfer_fee: float = 0.00001
    exchange_fee: float = 0.0000341


DEFAULT_SCHEDULES: tuple[FeeSchedule, ...] = (
    FeeSchedule(date(2008, 9, 19), stamp_duty_sell=0.001, transfer_fee=0.00002, exchange_fee=0.0000487),
    FeeSchedule(date(2022, 4, 29), stamp_duty_sell=0.001, transfer_fee=0.00001, exchange_fee=0.0000487),
    FeeSchedule(date(2023, 8, 28), stamp_duty_sell=0.0005, transfer_fee=0.00001, exchange_fee=0.0000341),
)


class FeeModel:
    def __init__(self, schedules: tuple[FeeSchedule, ...] = DEFAULT_SCHEDULES, **overrides: Any) -> None:
        if not schedules:
            raise ValueError("费率表为空")
        self.schedules = tuple(sorted((replace(s, **overrides) for s in schedules), key=lambda s: s.effective_from))

    def schedule_on(self, d: date) -> FeeSchedule:
        chosen = None
        for s in self.schedules:
            if s.effective_from <= d:
                chosen = s
        if chosen is None:
            raise ValueError(f"{d} 早于费率表起点")
        return chosen

    def cost(self, d: date, side: str, notional: float) -> float:
        if side not in ("buy", "sell"):
            raise ValueError(side)
        s = self.schedule_on(d)
        commission = max(notional * s.commission_rate, s.min_commission_cny) if notional > 0 else 0.0
        fee = commission + notional * (s.transfer_fee + s.exchange_fee)
        if side == "sell":
            fee += notional * s.stamp_duty_sell
        return round(fee, 2)
