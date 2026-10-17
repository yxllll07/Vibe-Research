"""交易系统 v1 的策略定义:M0 规则(DATA_R0_v1)+ 市场状态过滤 + 双槽仓位 + 固定退出。

所有参数在 fit_strategy 时一次性从训练期决定,写成 strategy.json(含规则、状态模型参数、可做状态、
仓位与退出参数、哈希),之后每天只读不改。状态模型以 JSON 保存高斯混合的参数,可审阅、可复现。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from ..candidates.modes import MODES, candidate_rows, pick_slots, top_k
from ..config import TZ, Config
from ..evaluation.experiment import apply_rules, load_dataset
from ..evaluation.proxy_backtest import HOLD_DAYS, STOP_PCT
from ..evaluation.regime import MIN_STATE_DAYS, RegimeModel, daily_top_n, fit_model
from ..features.market import MARKET_FEATURES

VERSION = "strategy_v1"
GATE_TOP_N = 5
PARAMS: dict[str, Any] = {
    "top_k_per_mode": 2,
    "position_pct": 0.10,        # 0.5% 权益风险 / 5% 止损 ≈ 10% 名义仓位
    "slot_cap": 0.50,            # 主动槽 / 核心槽各自上限
    "slippage_bps": 10.0,
    "hold_days": HOLD_DAYS,      # A/B/D 次日开盘卖;C/E 持有 5 个交易日
    "stop_pct": STOP_PCT,        # C/E 入场价 −5% 止损
    # 验证期对比:不加状态过滤 −4.8%,加过滤 −12.8% → 按事先约定以验证期为准,默认关闭;状态仍在计划里展示
    "regime_gate": False,
    "regime_top_n_for_gate": GATE_TOP_N,
    "regime_min_state_days": MIN_STATE_DAYS,
}


def strategy_dir(cfg: Config) -> Path:
    return cfg.data_root / "system" / VERSION


def market_table(feats: pd.DataFrame) -> pd.DataFrame:
    return feats.groupby("trade_date")[MARKET_FEATURES].first().reset_index()


def _gmm_to_json(m: RegimeModel) -> dict[str, Any]:
    g = m.gmm
    return {"features": MARKET_FEATURES, "mu": m.mu.tolist(), "sd": m.sd.tolist(), "weights": g.weights_.tolist(),
            "means": g.means_.tolist(), "covariances": g.covariances_.tolist(), "train_end": str(m.train_end)}


def _gmm_from_json(d: dict[str, Any]) -> RegimeModel:
    g = GaussianMixture(n_components=len(d["weights"]), covariance_type="diag")
    g.weights_ = np.array(d["weights"])
    g.means_ = np.array(d["means"])
    g.covariances_ = np.array(d["covariances"])
    g.precisions_cholesky_ = 1.0 / np.sqrt(g.covariances_)
    return RegimeModel(pd.Series(d["mu"], index=d["features"]), pd.Series(d["sd"], index=d["features"]), g,
                       date.fromisoformat(d["train_end"]))


@dataclass
class Strategy:
    spec: dict[str, Any]
    regime: RegimeModel

    @property
    def rules(self) -> dict[str, Any]:
        return self.spec["rules"]

    @property
    def hash(self) -> str:
        return self.spec["hash"]

    def tradeable(self, mode: str, state: int) -> bool:
        if not self.spec["params"].get("regime_gate", False):
            return True
        return int(state) in self.spec["tradeable_states"][mode]

    def gate_says(self, mode: str, state: int) -> bool:
        """训练期统计下该状态是否可做(无论开关是否打开,都在计划里展示)。"""
        return int(state) in self.spec["tradeable_states"][mode]

    @classmethod
    def load(cls, cfg: Config) -> "Strategy":
        spec = json.loads((strategy_dir(cfg) / "strategy.json").read_text(encoding="utf-8"))
        body = {k: v for k, v in spec.items() if k != "hash"}
        if _hash(body) != spec["hash"]:
            raise RuntimeError("strategy.json 被改动过(哈希不符);策略定义只能通过 system-fit 重新生成")
        return cls(spec, _gmm_from_json(spec["regime_model"]))


def _hash(body: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def fit_strategy(cfg: Config, run_dir: Path) -> Path:
    """只用训练期:规则来自 run_dir/rules.json(训练期推导),状态模型与可做状态也只看训练期。"""
    rules = json.loads((run_dir / "rules.json").read_text(encoding="utf-8"))
    data, feats, _ = load_dataset(cfg)
    train_end = date.fromisoformat(cfg["split"]["train"][1])
    data["score_m0"] = apply_rules(data, rules)
    regime = fit_model(market_table(feats), train_end)
    state = regime.predict(market_table(feats))
    tradeable: dict[str, list[int]] = {}
    gate_stats: dict[str, Any] = {}
    train = data[data["split"] == "train"]
    for mode in MODES:
        s = daily_top_n(train, "score_m0", mode, GATE_TOP_N).to_frame("net").join(state)
        by = s.groupby("state")["net"].agg(["mean", "count"])
        tradeable[mode] = [int(k) for k, r in by.iterrows() if r["mean"] > 0 and r["count"] >= MIN_STATE_DAYS]
        gate_stats[mode] = {str(int(k)): {"mean_net": float(r["mean"]), "days": int(r["count"])} for k, r in by.iterrows()}
    body = {
        "version": VERSION, "created_at": datetime.now(TZ).isoformat(timespec="seconds"),
        "source_run": run_dir.name, "train_period": cfg["split"]["train"], "config_hash": cfg.hash(),
        "rules": rules, "regime_model": _gmm_to_json(regime), "tradeable_states": tradeable,
        "gate_train_stats": gate_stats, "params": PARAMS,
        "notes": "规则、状态模型、可做状态全部来自训练期 2024-10-08 ~ 2025-09-30;每天只读不改。",
    }
    out = strategy_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    spec = {**body, "hash": _hash(body)}
    (out / "strategy.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return out / "strategy.json"


def make_picks(strategy: Strategy, feats: pd.DataFrame, days: list[date]) -> tuple[pd.DataFrame, pd.Series]:
    """对一组信号日生成交易计划:候选 → M0 打分 → 状态过滤 → 每模式前 k → 槽位。返回 (picks, 每日状态)。"""
    f = feats[feats["trade_date"].isin(days)]
    state = strategy.regime.predict(market_table(f))
    cands = candidate_rows(f, strategy.rules)
    cands["state"] = cands["trade_date"].map(state)
    ok = [strategy.tradeable(m, s) for m, s in zip(cands["mode"], cands["state"])]
    cands = cands[ok]
    pool = top_k(cands[["trade_date", "ts_code", "mode", "m0_score"]], "m0_score", k=strategy.spec["params"]["top_k_per_mode"])
    if pool.empty:
        return pool.assign(slot=pd.Series(dtype=object)), state
    slots = pick_slots(pool, "m0_score")[["trade_date", "ts_code", "mode", "slot"]]
    # 返回完整候选池(主选 + 备选),slot 只在被槽位选中的主选上有值
    return pool.merge(slots, on=["trade_date", "ts_code", "mode"], how="left"), state
