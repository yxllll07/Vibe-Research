"""M1 基线:逻辑回归 + LightGBM(不兼容时退回 sklearn HistGradientBoosting)。

每个模式单独建模,只在该模式的入池样本上训练;目标统一为「代理成交后扣费净收益为正」,
这样不同模式的分数是同一含义的概率,主动槽才能跨模式比较:
- A / B / D:T 日开盘买、T+1 日开盘卖(第一个可卖日)
- C / E:T 日开盘买、持有 5 个交易日(T+4 收盘)
两套特征做消融:stock(只看个股) vs full(个股 + 市场 + 行业)。
固定参数与随机种子,不在验证期调参;盲测期只在锁定后评估一次。
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from ..features.market import MARKET_FEATURES
from ..features.sector import SECTOR_FEATURES
from ..features.stock import STOCK_FEATURES
from .split import TrainOnlyScaler

SEED = 20261017
ROUND_TRIP_COST = 0.0015  # 目标阈值:约等于佣金 + 印花税 + 滑点的往返成本,只用来定义标签
TARGETS = {"A": "r_open_next_open", "B": "r_open_next_open", "D": "r_open_next_open", "C": "r_h5", "E": "r_h5"}
FEATURE_SETS = {
    "stock": STOCK_FEATURES,
    "full": STOCK_FEATURES + MARKET_FEATURES + SECTOR_FEATURES,
}


def add_trade_targets(labels: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """r_open_next_open:T 开盘 → T+1 开盘(调整后),由面板的 open / prev_close 连乘得到。"""
    p = feats[["trade_date", "ts_code", "open", "prev_close", "close"]].sort_values(["ts_code", "trade_date"])
    cum = np.log(p["close"] / p["prev_close"]).groupby(p["ts_code"]).cumsum()
    # 某日开盘在调整后坐标里的位置 = 前一日累计 + log(开盘 / 当日前收)
    p = p.assign(_open_log=cum.groupby(p["ts_code"]).shift(1).fillna(0) + np.log(p["open"] / p["prev_close"]))
    g = p.groupby("ts_code", sort=False)
    p["_open_log_t"] = g["_open_log"].shift(-1)
    p["_open_log_t1"] = g["_open_log"].shift(-2)
    p["r_open_next_open"] = np.exp(p["_open_log_t1"] - p["_open_log_t"]) - 1
    out = labels.merge(p[["trade_date", "ts_code", "r_open_next_open"]],
                       left_on=["signal_date", "ts_code"], right_on=["trade_date", "ts_code"], how="left")
    return out.drop(columns="trade_date")


def _clean(X: pd.DataFrame) -> pd.DataFrame:
    """转浮点,±inf(分母为 0 的比值)按缺失处理;缺失用训练期中位数填补(在调用处)。"""
    return X.astype(float).replace([np.inf, -np.inf], np.nan)


def _gbm():
    try:
        import lightgbm as lgb

        return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                  random_state=SEED, verbose=-1)
    except Exception:  # CPU / 预编译包不兼容时的后备
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, random_state=SEED)


@dataclass
class ModeModel:
    mode: str
    feature_set: str
    kind: str
    columns: list[str]
    model: object
    scaler: TrainOnlyScaler | None = None
    medians: pd.Series | None = None
    train_rows: int = 0
    train_pos_rate: float = float("nan")
    meta: dict = field(default_factory=dict)

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = _clean(df[self.columns])
        X = (X.fillna(self.medians) if self.medians is not None else X).fillna(0.0)
        return self.scaler.transform(X) if self.scaler is not None else X

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if df.empty:
            return np.array([])
        return self.model.predict_proba(self._X(df))[:, 1]  # type: ignore[attr-defined]


def fit_mode_model(train: pd.DataFrame, mode: str, feature_set: str, kind: str, train_end) -> ModeModel | None:
    cols = FEATURE_SETS[feature_set]
    y = (train[TARGETS[mode]] > ROUND_TRIP_COST).astype(int)
    if len(train) < 100 or y.nunique() < 2:
        return None
    X = _clean(train[cols])
    med = X.median()
    X = X.fillna(med).fillna(0.0)
    med = med.fillna(0.0)
    scaler = None
    if kind == "logit":
        scaler = TrainOnlyScaler(train_end).fit(X, train["trade_date"])
        model = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", random_state=SEED)
        model.fit(scaler.transform(X), y)
    else:
        model = _gbm()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
    return ModeModel(mode, feature_set, kind, cols, model, scaler, med, len(train), float(y.mean()))


def score_metrics(df: pd.DataFrame, score: np.ndarray, mode: str) -> dict:
    score = np.asarray(score, dtype=float)
    ok = ~np.isnan(score)
    unscored = int((~ok).sum())
    df, score = df[ok], score[ok]
    y = (df[TARGETS[mode]] > ROUND_TRIP_COST).astype(int).to_numpy()
    r = df[TARGETS[mode]].to_numpy()
    out = {"rows": int(len(df)), "unscored_rows": unscored, "base_rate": float(y.mean()) if len(y) else None}
    if len(df) and len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, score))
        out["brier"] = float(brier_score_loss(y, np.clip(score, 0, 1)))
    tmp = df.assign(_s=score, _y=y, _r=r)
    top = tmp.sort_values(["trade_date", "_s"], ascending=[True, False]).groupby("trade_date").head(1)
    out["days"] = int(tmp["trade_date"].nunique())
    out["precision_at_1"] = float(top["_y"].mean()) if len(top) else None
    out["top1_mean_target_return"] = float(np.nanmean(top["_r"])) if len(top) else None
    # 每日截面 RankIC 的均值
    ics = [g["_s"].rank().corr(g["_r"].rank()) for _, g in tmp.groupby("trade_date") if len(g) >= 5]
    out["rank_ic_mean"] = float(np.nanmean(ics)) if ics else None
    out["rank_ic_days"] = len(ics)
    ics_a = np.array([x for x in ics if not np.isnan(x)])
    out["rank_ic_t"] = float(ics_a.mean() / ics_a.std() * np.sqrt(len(ics_a))) if len(ics_a) > 2 and ics_a.std() > 0 else None
    # 信号层诊断:每日池内按分数五等分,最高组 − 最低组的平均目标收益(只报告,不参与选配置)
    q = tmp.groupby("trade_date")["_s"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)
                                                    if len(s) >= 10 else pd.Series(np.nan, index=s.index))
    if q.notna().any():
        daily = tmp.assign(_q=q).dropna(subset=["_q"]).groupby(["trade_date", "_q"])["_r"].mean().unstack()
        if 0 in daily and 4 in daily:
            spread = (daily[4] - daily[0]).dropna()
            out["q5_mean_return"] = float(daily[4].mean())
            out["q1_mean_return"] = float(daily[0].mean())
            out["q5_minus_q1"] = float(spread.mean())
            out["q5_minus_q1_t"] = float(spread.mean() / spread.std() * np.sqrt(len(spread))) if spread.std() > 0 else None
    return out


def model_hash(path) -> str:
    """对落盘的 models.pkl 文件字节取哈希(反序列化再序列化的字节不保证一致,不能对内存对象取)。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
