#!/usr/bin/env bash
# T 日 9:31 运行:记录竞价修复系统前 5 名在 9:30 后的现价,用来检验真实买入价比开盘价差多少。
# 定时示例(crontab,服务器时区为 Asia/Shanghai):  31 9 * * 1-5  cd /path/to/Vibe-Research && ashare_lab/scripts/fill.sh >> .local/ashare_lab/fill.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/../.."
T="${1:-$(TZ=Asia/Shanghai date +%F)}"
PY=.venv/bin/python
if ! $PY -c "
import sys; from datetime import date
from ashare_lab.config import load_config
from ashare_lab.data.calendar import TradingCalendar
from ashare_lab.data.ingest import calendar_path
sys.exit(0 if TradingCalendar.load(calendar_path(load_config().data_root)).is_open(date.fromisoformat('$T')) else 3)"; then
  echo "$T 不是交易日,跳过"; exit 0
fi
$PY -m ashare_lab relay-fill --date "$T"
