"""按日期切分、purge / embargo、只在训练期拟合的标准化器、盲测锁。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.calendar import TradingCalendar


class BlindTestLocked(RuntimeError):
    pass


@dataclass(frozen=True)
class Period:
    name: str
    start: date
    end: date


def periods_from_config(split_cfg: dict) -> list[Period]:
    out = []
    for name in ("train", "validation", "blind_test"):
        s, e = split_cfg[name]
        out.append(Period(name, date.fromisoformat(s), date.fromisoformat(e)))
    for a, b in zip(out, out[1:]):
        if not a.end < b.start:
            raise ValueError(f"{a.name} 与 {b.name} 重叠")
    return out


def assign_split(signal_dates: pd.Series, periods: list[Period], cal: TradingCalendar,
                 max_horizon: int, embargo: int) -> pd.Series:
    """给每个样本打所属期;标签窗口跨进下一期的样本清除(purge),每期开头 embargo 个交易日也清除。"""
    result = pd.Series(pd.NA, index=signal_dates.index, dtype="string")
    for i, p in enumerate(periods):
        in_p = (signal_dates >= p.start) & (signal_dates <= p.end)
        if i + 1 < len(periods):
            nxt = periods[i + 1].start
            first_next = next(d for d in cal.days if d >= nxt)
            # 标签用到 D 之后 max_horizon 个交易日;最后一个标签日必须早于下一期第一个交易日
            cutoff = cal.shift(first_next, -max_horizon - 1)
            in_p &= signal_dates <= cutoff
        if i > 0:
            first_day = next(d for d in cal.days if d >= p.start)
            in_p &= signal_dates >= cal.shift(first_day, embargo)
        result[in_p] = p.name
    return result


class TrainOnlyScaler:
    """只允许用训练期样本拟合;拟合数据里出现训练期之后的日期直接报错。"""

    def __init__(self, train_end: date) -> None:
        self.train_end = train_end
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None
        self.fit_range: tuple[date, date] | None = None

    def fit(self, X: pd.DataFrame, signal_dates: pd.Series) -> "TrainOnlyScaler":
        if (signal_dates > self.train_end).any():
            raise ValueError(f"标准化器只能用 {self.train_end} 及以前的训练样本拟合")
        self.mean_ = X.mean()
        self.std_ = X.std(ddof=0).replace(0, 1.0)
        self.fit_range = (signal_dates.min(), signal_dates.max())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("先 fit 再 transform")
        return (X - self.mean_) / self.std_


class BlindTestLock:
    """盲测期标签只有在模型与配置哈希锁定后才能读一次;揭盲后再改配置,盲测作废。"""

    def __init__(self, lock_file: Path) -> None:
        self.lock_file = lock_file

    def lock(self, model_hash: str, config_hash: str) -> None:
        if self.lock_file.exists():
            raise BlindTestLocked(f"{self.lock_file} 已存在,不能重复锁定")
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file.write_text(json.dumps({"model_hash": model_hash, "config_hash": config_hash}), encoding="utf-8")

    def assert_unlocked_for(self, model_hash: str, config_hash: str) -> None:
        if not self.lock_file.exists():
            raise BlindTestLocked("模型与配置未锁定,禁止读取盲测期标签")
        rec = json.loads(self.lock_file.read_text(encoding="utf-8"))
        if rec != {"model_hash": model_hash, "config_hash": config_hash}:
            raise BlindTestLocked("当前模型或配置与锁定时不一致,盲测已不合法")


def as_float_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].astype(float).to_numpy()
