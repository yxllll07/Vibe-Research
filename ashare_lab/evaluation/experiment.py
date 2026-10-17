"""M0 / M1 实验:训练期推导规则与拟合模型,验证期评估与代理回测;盲测期只在锁定后由 evaluate_blind 评估一次。

产出目录 <数据根>/experiments/<experiment_id>/<run_id>/:
config.yaml、dataset_manifest.json、rules.json、metrics.json、ablation.csv、daily_predictions.parquet、
trades_<配置>.csv、equity_<配置>.csv、models.pkl、report.md;揭盲后另有 blind_metrics.json、blind_report.md
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
from ..candidates.modes import MODES, candidate_rows, pick_slots, top_k
from ..config import TZ, Config
from ..data.calendar import TradingCalendar
from ..data.ingest import calendar_path
from . import proxy_backtest
from .models import FEATURE_SETS, TARGETS, ModeModel, add_trade_targets, fit_mode_model, model_hash, score_metrics
from .rule_mining import dump, mine_rules, rule_scores
from .split import BlindTestLock, Period, assign_split, periods_from_config
from .stats import block_bootstrap_ci, daily_returns

log = logging.getLogger(__name__)
MAX_HORIZON = 6  # C/E 在 T+5 开盘卖出 = D 之后第 6 个交易日
CONFIGS = ["m0"] + [f"m1_{k}_{fs}" for k in ("logit", "gbm") for fs in FEATURE_SETS]
CAPITAL = 1_000_000


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


def apply_rules(data: pd.DataFrame, rules: dict) -> pd.Series:
    out = pd.Series(float("nan"), index=data.index)
    for mode in MODES:
        idx = data["mode"] == mode
        out[idx] = rule_scores(data[idx], rules)[mode].values
    return out


def score_with_models(data: pd.DataFrame, models: list[ModeModel], name: str) -> pd.Series:
    kind, fs = name.split("_")[1:]
    out = pd.Series(float("nan"), index=data.index)
    for m in models:
        if m.kind == kind and m.feature_set == fs:
            idx = data["mode"] == m.mode
            if idx.any():
                out[idx] = m.predict(data[idx])
    return out


def backtest_configs(data: pd.DataFrame, feats: pd.DataFrame, cal: TradingCalendar, period: Period,
                     cfg: Config) -> dict[str, proxy_backtest.BacktestResult]:
    res = {}
    for name in CONFIGS:
        col = f"score_{name}"
        sub = data.dropna(subset=[col])
        pool = top_k(sub[["trade_date", "ts_code", "mode", col]], col, k=cfg["pool"]["max_candidates_per_mode"])
        picks = pick_slots(pool, col)
        r = proxy_backtest.run(picks, feats, cal, period.start, period.end, capital=CAPITAL,
                               slippage_bps=cfg["execution"]["slippage_bps"])
        r.metrics["daily_return_bootstrap"] = block_bootstrap_ci(daily_returns(r.equity, CAPITAL))
        res[name] = r
    m0_daily = daily_returns(res["m0"].equity, CAPITAL)
    for name, r in res.items():
        if name != "m0":
            diff = daily_returns(r.equity, CAPITAL) - m0_daily
            r.metrics["vs_m0_daily_diff_bootstrap"] = block_bootstrap_ci(diff)
    return res


def run_experiment(cfg: Config) -> Path:
    data, feats, cal = load_dataset(cfg)
    periods = {p.name: p for p in periods_from_config(cfg["split"])}
    tr, va = periods["train"], periods["validation"]
    train = data[(data["split"] == "train") & _fillable(data)]
    if train.empty or (data["split"] == "validation").sum() == 0:
        raise RuntimeError("训练或验证期没有样本:先 ingest + build 覆盖到验证期结束")
    manifest = json.loads((features_dir(cfg) / "manifest.json").read_text(encoding="utf-8"))

    log.info("训练期推导 M0 规则")
    rules = mine_rules(train, tr.start, tr.end, manifest["raw_inputs_sha256_digest"])
    data["score_m0"] = apply_rules(data, rules)
    train = data[(data["split"] == "train") & _fillable(data)]

    models: list[ModeModel] = []
    metrics_rows = []
    valid = data[data["split"] == "validation"]
    for kind in ("logit", "gbm"):
        for fs in FEATURE_SETS:
            name = f"m1_{kind}_{fs}"
            for mode in MODES:
                tm = train[(train["mode"] == mode) & train[TARGETS[mode]].notna()]
                mm = fit_mode_model(tm, mode, fs, kind, tr.end)
                if mm is None:
                    log.warning("%s %s 样本不足,跳过", name, mode)
                    continue
                models.append(mm)
            data[f"score_{name}"] = score_with_models(data, models, name)
    valid = data[data["split"] == "validation"]
    for name in CONFIGS:
        for mode in MODES:
            vm = valid[(valid["mode"] == mode) & valid[TARGETS[mode]].notna()]
            metrics_rows.append({"config": name, "mode": mode,
                                 **score_metrics(vm, vm[f"score_{name}"].to_numpy(dtype=float), mode)})
    log.info("验证期代理回测")
    backtests = backtest_configs(valid, feats, cal, va, cfg)

    run_id = datetime.now(TZ).strftime("%Y%m%dT%H%M%S")
    out = cfg.data_root / "experiments" / cfg["experiment_id"] / run_id
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg.path, out / "config.yaml")
    shutil.copy(features_dir(cfg) / "manifest.json", out / "dataset_manifest.json")
    (out / "rules.json").write_text(dump(rules), encoding="utf-8")
    ablation = pd.DataFrame(metrics_rows)
    ablation.to_csv(out / "ablation.csv", index=False)
    keep = ["trade_date", "ts_code", "mode", "split"] + [f"score_{c}" for c in CONFIGS]
    data.loc[data["split"] != "blind_test", keep].to_parquet(out / "daily_predictions.parquet", index=False)
    for name, res in backtests.items():
        res.trades.to_csv(out / f"trades_{name}.csv", index=False)
        res.equity.to_csv(out / f"equity_{name}.csv", index=False)
    with (out / "models.pkl").open("wb") as fh:
        pickle.dump(models, fh)
    metrics = {
        "run_id": run_id, "rule_version": rules["rule_version"], "config_hash": cfg.hash(),
        "model_hash": model_hash(out / "models.pkl"),
        "periods": {k: [str(v.start), str(v.end)] for k, v in periods.items()},
        "samples": {s: int((data["split"] == s).sum()) for s in ("train", "validation", "blind_test")},
        "blind_test_evaluated": False,
        "rules_chosen": {m: [(c["feature"], c["sign"], round(c["t"], 2)) for c in rules["modes"][m]["chosen"]]
                         for m in MODES},
        "backtests_validation": {k: v.metrics for k, v in backtests.items()},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text(render_report(metrics, ablation, "验证期"), encoding="utf-8")
    return out


def _pct(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:.2f}%"


def _num(x, nd=3) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def _ci(b: dict | None) -> str:
    if not b or not b.get("ci"):
        return "—"
    return f"{b['mean'] * 1e4:.1f} [{b['ci'][0] * 1e4:.1f}, {b['ci'][1] * 1e4:.1f}]"


def render_report(m: dict, ablation: pd.DataFrame, period_name: str, backtests_key: str = "backtests_validation") -> str:
    L = [f"# 实验报告 {m['run_id']}({period_name})", "",
         f"> ⚠️ {proxy_backtest.LABEL}。", f"> M0 规则版本 **{m['rule_version']}**:只用训练期数据推导(见 rules.json)。", "",
         f"- 配置哈希 `{m['config_hash']}` · 模型哈希 `{m['model_hash']}`",
         f"- 切分:{m['periods']}", f"- 候选行:{m['samples']}", "",
         "## M0 规则(训练期 RankIC 选出的特征,方向 · t 值)", ""]
    for mode, chosen in m["rules_chosen"].items():
        L.append(f"- {mode}:" + ("、".join(f"{f}({'+' if s > 0 else '−'}, t={t})" for f, s, t in chosen) or "无稳定特征 → 不出票"))
    L += ["", f"## {period_name}代理回测", "",
          "| 配置 | 交易 | 未成交买单 | 胜率 | 平均每笔净收益 | 总收益 | 最大回撤 | 夏普 | 日均收益 bp [95% CI] | 相对 M0 日均 bp [95% CI] |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, b in m[backtests_key].items():
        L.append(f"| {name} | {b.get('trades', 0)} | {b.get('unfilled_buys', 0)} | {_pct(b.get('win_rate'))} | "
                 f"{_pct(b.get('avg_net_return_per_trade'))} | {_pct(b.get('total_return'))} | "
                 f"{_pct(b.get('max_drawdown'))} | {_num(b.get('sharpe_daily_annualized'), 2)} | "
                 f"{_ci(b.get('daily_return_bootstrap'))} | {_ci(b.get('vs_m0_daily_diff_bootstrap'))} |")
    L += ["", "### 分模式(按交易)", ""]
    for name, b in m[backtests_key].items():
        bm = b.get("by_mode", {})
        L.append(f"- {name}:" + ";".join(f"{k} {v['trades']} 笔 胜率 {_pct(v['win_rate'])} 均 {_pct(v['avg_net_return'])}"
                                          for k, v in bm.items()))
    L += ["", "### 未成交原因", ""]
    for name, b in m[backtests_key].items():
        L.append(f"- {name}:{b.get('unfilled_by_reason', {})}")
    if ablation is not None and not ablation.empty:
        L += ["", f"## 分模式打分质量({period_name},目标 = 扣费后代理收益为正)", "",
              "| 配置 | 模式 | 样本 | 正例率 | AUC | Precision@1 | Top1 平均目标收益 | RankIC 均值 (t) | 五分组 Q5−Q1 (t) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in ablation.sort_values(["mode", "config"]).itertuples(index=False):
            g = lambda k: getattr(r, k, None)  # noqa: E731
            L.append(f"| {r.config} | {r.mode} | {r.rows} | {_pct(r.base_rate)} | {_num(g('auc'))} | "
                     f"{_pct(r.precision_at_1)} | {_pct(r.top1_mean_target_return)} | "
                     f"{_num(r.rank_ic_mean)} ({_num(g('rank_ic_t'), 1)}) | {_pct(g('q5_minus_q1'))} ({_num(g('q5_minus_q1_t'), 1)}) |")
    L += ["", "## 解读边界", "",
          "- 5 个配置同时比较,存在多重检验:单个配置的置信区间不跨 0 也不足以单独下结论。",
          "- 代理成交按开盘价 + 滑点,开盘即涨停 / 一字板不算买到;没有分钟数据,无法确认开盘后真实可成交量。",
          "- 换手率 / 市值特征缺失;开板次数、首封时间 UNKNOWN。", ""]
    return "\n".join(L)


def lock_for_blind_test(run_dir: Path) -> Path:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    lock = BlindTestLock(run_dir / "BLIND_TEST_LOCK.json")
    lock.lock(metrics["model_hash"], metrics["config_hash"])
    return lock.lock_file


def evaluate_blind(cfg: Config, run_dir: Path) -> Path:
    """只评估一次:锁定文件里的模型 / 配置哈希必须与当前一致;已评估过则拒绝。"""
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if (run_dir / "blind_metrics.json").exists():
        raise RuntimeError("盲测期已评估过一次;再次评估不合法")
    mh = model_hash(run_dir / "models.pkl")
    if mh != metrics["model_hash"]:
        raise RuntimeError("models.pkl 与 metrics.json 里的模型哈希不一致")
    BlindTestLock(run_dir / "BLIND_TEST_LOCK.json").assert_unlocked_for(mh, cfg.hash())
    with (run_dir / "models.pkl").open("rb") as fh:
        models: list[ModeModel] = pickle.load(fh)
    rules = json.loads((run_dir / "rules.json").read_text(encoding="utf-8"))
    data, feats, cal = load_dataset(cfg)
    bt = data[data["split"] == "blind_test"].copy()
    bt["score_m0"] = apply_rules(bt, rules)
    for name in CONFIGS[1:]:
        bt[f"score_{name}"] = score_with_models(bt, models, name)
    period = {p.name: p for p in periods_from_config(cfg["split"])}["blind_test"]
    rows = []
    for name in CONFIGS:
        for mode in MODES:
            vm = bt[(bt["mode"] == mode) & bt[TARGETS[mode]].notna()]
            rows.append({"config": name, "mode": mode, **score_metrics(vm, vm[f"score_{name}"].to_numpy(dtype=float), mode)})
    res = backtest_configs(bt, feats, cal, period, cfg)
    blind = {**metrics, "blind_test_evaluated": True, "evaluated_at": datetime.now(TZ).isoformat(timespec="seconds"),
             "backtests_blind": {k: v.metrics for k, v in res.items()}}
    (run_dir / "blind_metrics.json").write_text(json.dumps(blind, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(rows).to_csv(run_dir / "blind_ablation.csv", index=False)
    for name, r in res.items():
        r.trades.to_csv(run_dir / f"blind_trades_{name}.csv", index=False)
        r.equity.to_csv(run_dir / f"blind_equity_{name}.csv", index=False)
    (run_dir / "blind_report.md").write_text(
        render_report(blind, pd.DataFrame(rows), "盲测期(只评估一次)", "backtests_blind"), encoding="utf-8")
    return run_dir / "blind_report.md"

