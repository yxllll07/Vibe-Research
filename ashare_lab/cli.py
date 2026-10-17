"""命令行入口:python -m ashare_lab <命令>。"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date

from .config import load_config


def _d(s: str) -> date:
    return date.fromisoformat(s if "-" in s else f"{s[:4]}-{s[4:6]}-{s[6:]}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ashare_lab", description="A-Structure Lab 研究工具(不下单)")
    p.add_argument("--config", default=None, help="配置文件,默认 configs/eod_v1.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("env-report", help="扫描运行环境,写 ENVIRONMENT_REPORT.md")
    sub.add_parser("calendar", help="拉取交易日历")
    sub.add_parser("sectors", help="拉取申万 2021 行业个股归属及变迁")
    ing = sub.add_parser("ingest", help="逐交易日拉取通达信盘后包并归一化(可续传)")
    ing.add_argument("--start", default=None)
    ing.add_argument("--end", default=None)
    ing.add_argument("--force-normalize", action="store_true", help="已有归一化文件也重算")
    b = sub.add_parser("build", help="构建特征表与标签表")
    b.add_argument("--start", default=None)
    b.add_argument("--end", default=None)
    sub.add_parser("experiment", help="M0/M1 训练 + 验证期评估与代理回测(不碰盲测期)")
    lk = sub.add_parser("lock", help="锁定某次实验的模型与配置哈希,允许之后评估一次盲测期")
    lk.add_argument("--run", default=None, help="实验目录,默认最新一次")
    bl = sub.add_parser("blind", help="对已锁定的实验评估盲测期(只允许一次)")
    bl.add_argument("--run", default=None, help="实验目录,默认最新一次")
    fz = sub.add_parser("freeze", help="为信号日 D 写冻结票池 frozen.vN.json")
    fz.add_argument("--signal-date", required=True)
    fz.add_argument("--score", default="m1_gbm_full", help="m0 / m1_logit_stock / m1_logit_full / m1_gbm_stock / m1_gbm_full")
    fz.add_argument("--replay", action="store_true", help="历史复现:freeze_at 记为 D 日 21:00 并打标")
    q = sub.add_parser("quality", help="数据覆盖率与一致性报告")
    q.add_argument("--start", default=None)
    q.add_argument("--end", default=None)
    q.add_argument("--crosscheck-date", action="append", default=[], help="与东财涨停池交叉核对的日期,可多次")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)

    if args.cmd == "env-report":
        from .env_report import write_report

        print(write_report(cfg))
    elif args.cmd == "calendar":
        from .data.ingest import ingest_calendar

        df = ingest_calendar(cfg)
        print(f"交易日 {len(df)} 个:{df.trade_date.iloc[0]} ~ {df.trade_date.iloc[-1]}")
    elif args.cmd == "sectors":
        from .data.ingest import ingest_sectors

        m = ingest_sectors(cfg)
        print(f"申万成员记录 {len(m)} 条,股票 {m.ts_code.nunique()} 只,计入日期 {m.effective_from.min()} ~ {m.effective_from.max()}")
    elif args.cmd == "ingest":
        from .data.ingest import ingest_tdx

        start = _d(args.start or cfg["data"]["ingest_start"])
        end = _d(args.end or cfg["data"]["ingest_end"])
        rep = ingest_tdx(cfg, start, end, force_normalize=args.force_normalize)
        print(json.dumps(rep.to_json(), ensure_ascii=False, indent=2))
        return 0 if not rep.failed else 1
    elif args.cmd == "build":
        from .build import build

        out = build(cfg, _d(args.start or cfg["data"]["ingest_start"]), _d(args.end or cfg["data"]["ingest_end"]))
        print((out / "manifest.json").read_text(encoding="utf-8"))
    elif args.cmd == "experiment":
        from .evaluation.experiment import run_experiment

        print(run_experiment(cfg))
    elif args.cmd in ("lock", "blind"):
        from pathlib import Path

        from .candidates.freeze_job import latest_run
        from .evaluation.experiment import evaluate_blind, lock_for_blind_test

        run = Path(args.run) if args.run else latest_run(cfg)
        print(lock_for_blind_test(run) if args.cmd == "lock" else evaluate_blind(cfg, run))
    elif args.cmd == "freeze":
        from .candidates.freeze_job import freeze

        print(freeze(cfg, _d(args.signal_date), args.score, replay=args.replay))
    elif args.cmd == "quality":
        from .data.quality import run_quality

        start = _d(args.start or cfg["data"]["ingest_start"])
        end = _d(args.end or cfg["data"]["ingest_end"])
        path = run_quality(cfg, start, end, [_d(x) for x in args.crosscheck_date])
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
