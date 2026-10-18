#!/usr/bin/env bash
# T 日 9:26 ~ 9:29 运行:竞价修复系统拉全市场竞价出前 5;情绪开关系统只抓昨晚预选池的竞价(对照)。
# 用法:ashare_lab/scripts/auction.sh [YYYY-MM-DD]   默认今天(北京时间)
# 定时示例(crontab,服务器时区为 Asia/Shanghai):  26 9 * * 1-5  cd /path/to/Vibe-Research && ashare_lab/scripts/auction.sh >> .local/ashare_lab/auction.log 2>&1
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
# 竞价修复系统(主)先跑,情绪开关系统(对照)后跑;任一失败不影响另一个
cat "$($PY -m ashare_lab relay-pick --date "$T")" || echo "relay-pick 失败"
out=$($PY -m ashare_lab auction-v3 --date "$T") && { cat "${out%.json}.md" 2>/dev/null || cat "$out"; } || echo "auction-v3 失败"
