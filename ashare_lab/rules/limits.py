"""按日期生效的涨跌幅限制与涨跌停价。

规则来源(写进 docs/DATA_AUDIT.md,改动要连同测试一起改):
- 主板普通股票 ±10%。
- 主板风险警示股票:2026-07-06 前 ±5%;沪深交易所 2026-04-24 发布的修订版交易规则自 2026-07-06 起调为 ±10%。
- 创业板 / 科创板 ±20%,北交所 ±30%(本项目不交易,仅用于全市场广度统计)。
- 注册制新股上市前 5 个交易日不设涨跌幅限制(主板自 2023-04-10 全面注册制起适用)。
  之前的主板新股:上市首日按发行价 +44% / −36% 的特殊价格笼子(不是常规涨跌停,这里按「无常规涨跌停」处理),
  第 2 个交易日起即 ±10%。
- 创业板 2020-08-24 注册制改革起 ±20%(本项目窗口都在此之后)。
- 退市整理期首个交易日不设涨跌幅限制(实测 21 条退市整理首日收盘全部越出 ±10%,与规则一致)。
涨跌停价 = 前收盘 × (1 ± 比例),四舍五入到 0.01 元;前收盘用交易所当日前收(除权日为除权参考价),
不能用上一日的原始收盘价,也不能用前复权价。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

MAIN_ST_10PCT_FROM = date(2026, 7, 6)
MAIN_REGISTRATION_FROM = date(2023, 4, 10)
REGISTRATION_IPO_NO_LIMIT_DAYS = 5
TICK = Decimal("0.01")


@dataclass(frozen=True)
class LimitBand:
    pct: float | None  # None = 当日无涨跌幅限制
    up: float | None
    down: float | None

    @property
    def has_limit(self) -> bool:
        return self.pct is not None


def limit_pct(board: str, is_risk_warning: bool, trade_date: date, listed_trading_days: int | None = None,
              delisting_first_day: bool = False) -> float | None:
    """listed_trading_days:上市第几个交易日(首日 = 1);未知时传 None(视为早已上市)。"""
    if listed_trading_days is not None and listed_trading_days <= REGISTRATION_IPO_NO_LIMIT_DAYS:
        if not (board == "main" and trade_date < MAIN_REGISTRATION_FROM):
            return None
        if listed_trading_days == 1:
            return None
    if delisting_first_day:
        return None
    if board == "main":
        if is_risk_warning and trade_date < MAIN_ST_10PCT_FROM:
            return 0.05
        return 0.10
    if board in ("chinext", "star"):
        return 0.20
    if board == "bse":
        return 0.30
    raise ValueError(f"未知板块 {board!r}")


def round_price(x: Decimal) -> float:
    return float(x.quantize(TICK, rounding=ROUND_HALF_UP))


def band_prices(prev_close: float, pct: float) -> tuple[float, float]:
    """(涨停价, 跌停价):前收 × (1 ± 比例),四舍五入到分。"""
    if not prev_close or prev_close <= 0:
        raise ValueError(f"前收盘 {prev_close!r} 无效,算不出涨跌停价")
    p = Decimal(str(prev_close))
    r = Decimal(str(pct))
    return round_price(p * (1 + r)), round_price(p * (1 - r))


def limit_band(prev_close: float, board: str, is_risk_warning: bool, trade_date: date,
               listed_trading_days: int | None = None) -> LimitBand:
    pct = limit_pct(board, is_risk_warning, trade_date, listed_trading_days)
    if pct is None:
        return LimitBand(None, None, None)
    up, down = band_prices(prev_close, pct)
    return LimitBand(pct, up, down)


def at_price(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and abs(a - b) < 1e-6
