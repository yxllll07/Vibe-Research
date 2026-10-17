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
    ing = sub.add_parser("ingest", help="逐交易日拉取通达信盘后包并归一化(可续传)")
    ing.add_argument("--start", default=None)
    ing.add_argument("--end", default=None)
    ing.add_argument("--force-normalize", action="store_true", help="已有归一化文件也重算")
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
    elif args.cmd == "ingest":
        from .data.ingest import ingest_tdx

        start = _d(args.start or cfg["data"]["ingest_start"])
        end = _d(args.end or cfg["data"]["ingest_end"])
        rep = ingest_tdx(cfg, start, end, force_normalize=args.force_normalize)
        print(json.dumps(rep.to_json(), ensure_ascii=False, indent=2))
        return 0 if not rep.failed else 1
    elif args.cmd == "quality":
        from .data.quality import run_quality

        start = _d(args.start or cfg["data"]["ingest_start"])
        end = _d(args.end or cfg["data"]["ingest_end"])
        path = run_quality(cfg, start, end, [_d(x) for x in args.crosscheck_date])
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
