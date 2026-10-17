"""M0 / M1 实验:训练期拟合、验证期评估与代理回测;盲测期只在锁定后由 evaluate_blind 评估一次。

产出目录 <数据根>/experiments/<experiment_id>/<run_id>/:
config.yaml、dataset_manifest.json、metrics.json、ablation.csv、daily_predictions.parquet、
trades_<配置>.csv、equity_<配置>.csv、models.pkl、report.md
"""
from __future__ import annotations

import json
import logging
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..build import features_dir
from ..candidates.modes import MODES, RULE_VERSION, candidate_rows, pick_slots, top_k
from ..config import TZ, Config
from ..data.calendar import TradingCalendar
from ..data.ingest import calendar_path
from . import proxy_backtest
from .models import FEATURE_SETS, TARGETS, add_trade_targets, fit_mode_model, model_hash, score_metrics
from .split import BlindTestLock, periods_from_config, assign_split

log = logging.getLogger(__name__)
MAX_HORIZON = 6  # C/E 在 T+5 开盘卖出 = D 之后第 6 个交易日
CONFIGS = ["m0"] + [f"m1_{k}_{fs}" for k in ("logit", "gbm") for fs in FEATURE_SETS]


def load_dataset(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, TradingCalendar]:
    fdir = features_dir(cfg)
    feats = pd.read_parquet(fdir / "features.parquet")
    labels = add_trade_targets(pd.read_parquet(fdir / "labels.parquet"), feats)
    cal = TradingCalendar.load(calendar_path(cfg.data_root))
    cands = candidate_rows(feats)
    data = cands.merge(feats, on=["trade_date", "ts_code"], how="left").merge(
        labels, left_on=["trade_date", "ts_code"], right_on=["signal_date", "ts_code"], how="left")
    periods = periods_from_config(cfg["split"])
    data["split"] = assign_split(data["trade_date"], periods, cal, MAX_HORIZON, cfg["split"]["embargo_trading_days"])
    return data, feats, cal


def _fillable(df: pd.DataFrame) -> pd.Series:
    """训练目标只用 T 日可买入的样本:一字涨停 / 停牌买不到,结果不能当成收益学。"""
    return ~df["t_one_word_up"].astype("boolean").fillna(False) & ~df["t_suspended"].astype("boolean").fillna(False)


def run_experiment(cfg: Config) -> Path:
    data, feats, cal = load_dataset(cfg)
    periods = {p.name: p for p in periods_from_config(cfg["split"])}
    tr, va = periods["train"], periods["validation"]
    train = data[(data["split"] == "train") & _fillable(data)]
    valid = data[data["split"] == "validation"]
    if valid.empty or train.empty:
        raise RuntimeError("训练或验证期没有样本:先 ingest + build 覆盖到验证期结束")

    models, metrics_rows = [], []
    data["score_m0"] = data["m0_score"]
    for kind in ("logit", "gbm"):
        for fs in FEATURE_SETS:
            name = f"m1_{kind}_{fs}"
            data[f"score_{name}"] = pd.NA
            for mode in MODES:
                tm = train[(train["mode"] == mode) & train[TARGETS[mode]].notna()]
                mm = fit_mode_model(tm, mode, fs, kind, tr.end)
                if mm is None:
                    log.warning("%s %s 样本不足,跳过", name, mode)
                    continue
                models.append(mm)
                idx = data["mode"] == mode
                data.loc[idx, f"score_{name}"] = mm.predict(data[idx])
                vm = valid[(valid["mode"] == mode) & valid[TARGETS[mode]].notna()]
                met = score_metrics(vm, data.loc[vm.index, f"score_{name}"].astype(float).to_numpy(), mode)
                metrics_rows.append({"config": name, "mode": mode, "train_rows": mm.train_rows,
                                     "train_pos_rate": mm.train_pos_rate, **met})
    for mode in MODES:  # M0 规则分在验证期的同口径指标
        vm = valid[(valid["mode"] == mode) & valid[TARGETS[mode]].notna()]
        metrics_rows.append({"config": "m0", "mode": mode, **score_metrics(vm, vm["m0_score"].to_numpy(), mode)})

    backtests = {}
    vdata = data[data["split"] == "validation"]
    for name in CONFIGS:
        col = f"score_{name}"
        sub = vdata.dropna(subset=[col]).astype({col: float})
        pool = top_k(sub[["trade_date", "ts_code", "mode", col]], col, k=cfg["pool"]["max_candidates_per_mode"])
        picks = pick_slots(pool, col)
        res = proxy_backtest.run(picks, feats, cal, va.start, va.end,
                                 slippage_bps=cfg["execution"]["slippage_bps"])
        backtests[name] = res

    run_id = datetime.now(TZ).strftime("%Y%m%dT%H%M%S")
    out = cfg.data_root / "experiments" / cfg["experiment_id"] / run_id
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg.path, out / "config.yaml")
    shutil.copy(features_dir(cfg) / "manifest.json", out / "dataset_manifest.json")
    mh = model_hash(models)
    ablation = pd.DataFrame(metrics_rows)
    ablation.to_csv(out / "ablation.csv", index=False)
    keep = ["trade_date", "ts_code", "mode", "split"] + [f"score_{c}" for c in CONFIGS]
    data[keep].to_parquet(out / "daily_predictions.parquet", index=False)
    for name, res in backtests.items():
        res.trades.to_csv(out / f"trades_{name}.csv", index=False)
        res.equity.to_csv(out / f"equity_{name}.csv", index=False)
    with (out / "models.pkl").open("wb") as fh:
        pickle.dump(models, fh)
    metrics = {
        "run_id": run_id, "rule_version": RULE_VERSION, "config_hash": cfg.hash(), "model_hash": mh,
        "periods": {k: [str(v.start), str(v.end)] for k, v in periods.items()},
        "samples": {s: int((data["split"] == s).sum()) for s in ("train", "validation", "blind_test")},
        "blind_test_evaluated": False,
        "backtests_validation": {k: v.metrics for k, v in backtests.items()},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text(render_report(metrics, ablation), encoding="utf-8")
    return out


def _pct(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:.2f}%"


def _num(x, nd=3) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def render_report(m: dict, ablation: pd.DataFrame) -> str:
    L = [f"# 实验报告 {m['run_id']}", "",
         f"> ⚠️ {proxy_backtest.LABEL}。模式规则为 **{m['rule_version']}**(占位,待「五层核心选股 R1」替换)。",
         "> 盲测期(2026-04-01 ~ 2026-10-16)**未评估**;以下全部是验证期结果。", "",
         f"- 配置哈希 `{m['config_hash']}` · 模型哈希 `{m['model_hash']}`",
         f"- 切分:{m['periods']}", f"- 样本数(候选行):{m['samples']}", "",
         "## 验证期代理回测", "",
         "| 配置 | 交易 | 未成交买单 | 胜率 | 平均每笔净收益 | 总收益 | 最大回撤 | 夏普 |",
         "|---|---|---|---|---|---|---|---|"]
    for name, b in m["backtests_validation"].items():
        L.append(f"| {name} | {b.get('trades', 0)} | {b.get('unfilled_buys', 0)} | {_pct(b.get('win_rate'))} | "
                 f"{_pct(b.get('avg_net_return_per_trade'))} | {_pct(b.get('total_return'))} | "
                 f"{_pct(b.get('max_drawdown'))} | {_num(b.get('sharpe_daily_annualized'), 2)} |")
    L += ["", "## 未成交原因(验证期)", ""]
    for name, b in m["backtests_validation"].items():
        L.append(f"- {name}:{b.get('unfilled_by_reason', {})}")
    L += ["", "## 分模式打分质量(验证期,目标 = 扣费后代理收益为正)", "",
          "| 配置 | 模式 | 样本 | 正例率 | AUC | Brier | Precision@1 | Top1 平均目标收益 | RankIC 均值 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in ablation.sort_values(["mode", "config"]).itertuples(index=False):
        L.append(f"| {r.config} | {r.mode} | {r.rows} | {_pct(r.base_rate)} | {_num(getattr(r, 'auc', None))} | "
                 f"{_num(getattr(r, 'brier', None))} | {_pct(r.precision_at_1)} | {_pct(r.top1_mean_target_return)} | "
                 f"{_num(r.rank_ic_mean)} |")
    L += ["", "## 解读边界", "",
          "- 验证期只有约 6 个月;多个配置同时比较,存在多重检验问题,差异需用按交易日分块的自助法置信区间确认后才算数。",
          "- 代理成交按开盘价 + 滑点,开盘即涨停 / 一字板不算买到;但没有分钟数据,无法确认开盘后真实可成交量。",
          "- 换手率 / 市值特征缺失;开板次数、首封时间 UNKNOWN。", ""]
    return "\n".join(L)


def lock_for_blind_test(cfg: Config, run_dir: Path) -> Path:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    lock = BlindTestLock(run_dir / "BLIND_TEST_LOCK.json")
    lock.lock(metrics["model_hash"], metrics["config_hash"])
    return lock.lock_file
