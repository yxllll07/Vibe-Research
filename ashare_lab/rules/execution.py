"""日频代理成交(proxy_daily_only):只能用来检验信号方向,不代表 9:25 之后真的能成交。

保守口径(没有逐笔队列证据时一律不按乐观假设成交):
- 停牌:不成交。
- 买入:开盘即在涨停价(含一字板)→ 不成交;开盘价加滑点越过涨停价 → 不成交。
- 卖出:T+1,当日买入的股数当日不能卖;开盘即在跌停价 → 不成交,订单顺延;
  止损触发但当日一字跌停 → 卖不出,顺延到下一个可成交日,不按理论止损价记账。
- 100 股整数倍买入。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .fees import FeeModel
from .limits import LimitBand, at_price

LOT = 100


@dataclass(frozen=True)
class Bar:
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    prev_close: float
    volume_shares: int
    band: LimitBand

    @property
    def suspended(self) -> bool:
        return self.volume_shares == 0

    @property
    def one_word_up(self) -> bool:
        return self.band.has_limit and at_price(self.open, self.band.up) and at_price(self.low, self.band.up)

    @property
    def one_word_down(self) -> bool:
        return self.band.has_limit and at_price(self.open, self.band.down) and at_price(self.high, self.band.down)


@dataclass(frozen=True)
class Fill:
    trade_date: date
    ts_code: str
    side: str
    shares: int
    price: float
    fee: float


@dataclass(frozen=True)
class Rejection:
    trade_date: date
    ts_code: str
    side: str
    reason: str


@dataclass
class Lot:
    shares: int
    buy_date: date
    cost: float


@dataclass
class Broker:
    cash: float
    fees: FeeModel = field(default_factory=FeeModel)
    slippage_bps: float = 10.0
    positions: dict[str, list[Lot]] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)

    def _reject(self, d: date, code: str, side: str, reason: str) -> None:
        self.rejections.append(Rejection(d, code, side, reason))

    def shares_held(self, code: str) -> int:
        return sum(lot.shares for lot in self.positions.get(code, []))

    def sellable(self, code: str, d: date) -> int:
        return sum(lot.shares for lot in self.positions.get(code, []) if lot.buy_date < d)

    def buy_at_open(self, code: str, bar: Bar, budget_cny: float) -> Fill | None:
        d = bar.trade_date
        if bar.suspended:
            self._reject(d, code, "buy", "SUSPENDED")
            return None
        if bar.one_word_up:
            self._reject(d, code, "buy", "ONE_WORD_LIMIT_UP_NO_FILL")
            return None
        if bar.band.has_limit and at_price(bar.open, bar.band.up):
            self._reject(d, code, "buy", "OPEN_AT_LIMIT_UP_NO_QUEUE_EVIDENCE")
            return None
        slipped = bar.open * (1 + self.slippage_bps / 1e4)
        if bar.band.up is not None and slipped > bar.band.up + 1e-9:  # 先比未取整的价,取整会把越界单藏回界内
            self._reject(d, code, "buy", "SLIPPAGE_BEYOND_LIMIT_UP")
            return None
        price = round(slipped, 2)
        budget = min(budget_cny, self.cash)
        shares = int(budget // (price * LOT)) * LOT
        while shares > 0 and shares * price + self.fees.cost(d, "buy", shares * price) > self.cash:
            shares -= LOT
        if shares <= 0:
            self._reject(d, code, "buy", "BUDGET_BELOW_ONE_LOT")
            return None
        fee = self.fees.cost(d, "buy", shares * price)
        self.cash -= shares * price + fee
        self.positions.setdefault(code, []).append(Lot(shares, d, price))
        fill = Fill(d, code, "buy", shares, price, fee)
        self.fills.append(fill)
        return fill

    def _sell(self, code: str, d: date, shares: int, price: float) -> Fill:
        fee = self.fees.cost(d, "sell", shares * price)
        self.cash += shares * price - fee
        remaining = shares
        lots = self.positions[code]
        for lot in lots:  # 先进先出,只动已过 T+1 的批次
            if lot.buy_date >= d or remaining == 0:
                continue
            take = min(lot.shares, remaining)
            lot.shares -= take
            remaining -= take
        self.positions[code] = [lot for lot in lots if lot.shares > 0]
        if not self.positions[code]:
            del self.positions[code]
        fill = Fill(d, code, "sell", shares, price, fee)
        self.fills.append(fill)
        return fill

    def sell_at_open(self, code: str, bar: Bar, shares: int | None = None) -> Fill | None:
        d = bar.trade_date
        held = self.shares_held(code)
        if held == 0:
            self._reject(d, code, "sell", "NO_POSITION")
            return None
        avail = self.sellable(code, d)
        if avail == 0:
            self._reject(d, code, "sell", "T_PLUS_ONE")
            return None
        if bar.suspended:
            self._reject(d, code, "sell", "SUSPENDED")
            return None
        if bar.band.has_limit and at_price(bar.open, bar.band.down):
            self._reject(d, code, "sell", "OPEN_AT_LIMIT_DOWN_NO_EXIT")
            return None
        price = round(bar.open * (1 - self.slippage_bps / 1e4), 2)
        if bar.band.down is not None:
            price = max(price, bar.band.down)
        return self._sell(code, d, min(shares or avail, avail), price)

    def stop_loss(self, code: str, bar: Bar, stop_price: float) -> Fill | None:
        """日内触发止损:低点 ≤ 止损价才触发;跳空低开按开盘价,一字跌停卖不出。"""
        d = bar.trade_date
        if self.shares_held(code) == 0 or bar.low > stop_price:
            return None
        avail = self.sellable(code, d)
        if avail == 0:
            self._reject(d, code, "sell", "T_PLUS_ONE")
            return None
        if bar.suspended:
            self._reject(d, code, "sell", "SUSPENDED")
            return None
        if bar.one_word_down:
            self._reject(d, code, "sell", "LOCKED_LIMIT_DOWN_STOP_DEFERRED")
            return None
        if bar.band.has_limit and at_price(bar.open, bar.band.down):
            self._reject(d, code, "sell", "OPEN_AT_LIMIT_DOWN_NO_EXIT")
            return None
        raw = min(bar.open, stop_price)
        price = round(raw * (1 - self.slippage_bps / 1e4), 2)
        if bar.band.down is not None:
            price = max(price, bar.band.down)
        return self._sell(code, d, avail, price)
