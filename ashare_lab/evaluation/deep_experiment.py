"""M2 / M3 实验:与 M0 / M1 相同的候选、标签、切分、成本与回测口径。

盲测期已在 eod_v1 揭盲过一次,所以 2026-04-01 ~ 2026-10-16 的结果在报告里一律标为
「已揭盲期,非合法盲测」;判断以验证期为准。结构与超参数在研究方案里事先写定,没有据此调整。
"""
from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path

import pandas as pd

from ..config import Config
from . import deep
from .experiment import _fillable, apply_rules, backtest_configs, load_dataset, score_with_models
from .models import TARGETS, score_metrics
from .split import periods_from_config

log = logging.getLogger(__name__)
COMPARE = ["m0", "m1_gbm_stock", "m1_logit_full"] + deep.DEEP_CONFIGS


def run_deep(cfg: Config, run_dir: Path) -> Path:
    data, feats, cal = load_dataset(cfg)
    periods = {p.name: p for p in periods_from_config(cfg["split"])}
    rules = json.loads((run_dir / "rules.json").read_text(encoding="utf-8"))
    with (run_dir / "models.pkl").open("rb") as fh:
        models = pickle.load(fh)
    data["score_m0"] = apply_rules(data, rules)
    for name in ("m1_gbm_stock", "m1_logit_full"):
        data[f"score_{name}"] = score_with_models(data, models, name)

    t0 = time.time()
    T = deep.build_tensors(feats, periods["train"].end)
    log.info("张量 %s,用时 %.0fs", T.seq.shape, time.time() - t0)
    train = data[(data["split"] == "train") & _fillable(data)]
    has_target = pd.Series(False, index=train.index)
    for m in train["mode"].unique():
        idx = train["mode"] == m
        has_target[idx] = train.loc[idx, TARGETS[m]].notna()
    train = train[has_target]
    sample = deep.training_sample(train)
    log.info("深度模型训练样本 %d 行", len(sample))
    evalrows = data[data["split"].isin(["validation", "blind_test"])].reset_index(drop=True)
    info = {}
    for kind in deep.DEEP_CONFIGS:
        t0 = time.time()
        model = deep.fit(kind, T, sample)
        evalrows[f"score_{kind}"] = deep.predict(kind, model, T, evalrows)
        info[kind] = {"params": deep.n_params(model), "seconds": round(time.time() - t0)}
        log.info("%s 完成 %s", kind, info[kind])

    out: dict = {"configs": COMPARE, "deep_info": info, "train_samples": int(len(sample)), "periods": {}}
    tables = {}
    for split, label in (("validation", "验证期"), ("blind_test", "已揭盲期(2026-04 ~ 10,非合法盲测)")):
        sub = evalrows[evalrows["split"] == split]
        rows = []
        for name in COMPARE:
            for mode in sub["mode"].unique():
                vm = sub[(sub["mode"] == mode) & sub[TARGETS[mode]].notna()]
                rows.append({"config": name, "mode": mode,
                             **score_metrics(vm, vm[f"score_{name}"].to_numpy(dtype=float), mode)})
        tables[split] = pd.DataFrame(rows)
        bts = backtest_configs(sub, feats, cal, periods[split], cfg, configs=COMPARE)
        out["periods"][split] = {"label": label, "backtests": {k: v.metrics for k, v in bts.items()}}
    dst = run_dir / "deep"
    dst.mkdir(exist_ok=True)
    for split, t in tables.items():
        t.to_csv(dst / f"scores_{split}.csv", index=False)
    (dst / "deep_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (dst / "deep_report.md").write_text(render(out, tables), encoding="utf-8")
    return dst / "deep_report.md"


def _p(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:.2f}%"


def _n(x, nd=3) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def _ci(b) -> str:
    if not b or not b.get("ci"):
        return "—"
    return f"{b['mean'] * 1e4:.1f} [{b['ci'][0] * 1e4:.1f}, {b['ci'][1] * 1e4:.1f}]"


def render(out: dict, tables: dict) -> str:
    L = ["# M0 / M1 / M2 / M3 对比", "",
         "> ⚠️ 日频代理回测,不代表 9:25 之后可成交。M2 / M3 结构与超参数按研究方案事先写定,固定 4 个 epoch,未调参。",
         "> 2026-04 ~ 10 已在 eod_v1 揭盲过一次,这里的该段结果只作参考,**不是合法盲测**。", "",
         f"- 深度模型训练样本 {out['train_samples']} 行;参数量与用时:{out['deep_info']}", ""]
    for split, p in out["periods"].items():
        L += [f"## {p['label']}:代理回测", "",
              "| 配置 | 交易 | 胜率 | 平均每笔净收益 | 总收益 | 最大回撤 | 日均 bp [95% CI] | 相对 M0 日均 bp [95% CI] |",
              "|---|---|---|---|---|---|---|---|"]
        for name, b in p["backtests"].items():
            L.append(f"| {name} | {b.get('trades', 0)} | {_p(b.get('win_rate'))} | {_p(b.get('avg_net_return_per_trade'))} | "
                     f"{_p(b.get('total_return'))} | {_p(b.get('max_drawdown'))} | {_ci(b.get('daily_return_bootstrap'))} | "
                     f"{_ci(b.get('vs_m0_daily_diff_bootstrap'))} |")
        t = tables[split]
        L += ["", f"## {p['label']}:分模式 RankIC (t) / 五分组 Q5−Q1", "",
              "| 配置 | " + " | ".join(sorted(t["mode"].unique())) + " |", "|---" * (1 + t["mode"].nunique()) + "|"]
        for name in out["configs"]:
            cells = []
            for mode in sorted(t["mode"].unique()):
                r = t[(t.config == name) & (t["mode"] == mode)]
                if r.empty:
                    cells.append("—")
                    continue
                r = r.iloc[0]
                cells.append(f"{_n(r.get('rank_ic_mean'))} ({_n(r.get('rank_ic_t'), 1)}) / {_p(r.get('q5_minus_q1'))}")
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
    return "\n".join(L)
