#!/usr/bin/env bash
# 盘后日常:D 日收盘后运行(通达信盘后包通常 17:00 后发布)。
# 补日历和盘后包 → 重建特征(并出 v1 计划) → 更新申万行业 → 情绪 / 主线 / 龙头 → 当前系统(只用过去数据学开关表)的 T 日预选池。
# 用法:ashare_lab/scripts/daily.sh [YYYY-MM-DD]   默认今天(北京时间)
# 定时示例(crontab,服务器时区为 Asia/Shanghai):  30 20 * * 1-5  cd /path/to/Vibe-Research && ashare_lab/scripts/daily.sh >> .local/ashare_lab/daily.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${1:-$(TZ=Asia/Shanghai date +%F)}"
PY=.venv/bin/python
if ! $PY -c "
import sys; from datetime import date
from ashare_lab.config import load_config
from ashare_lab.data.ingest import ingest_calendar
df = ingest_calendar(load_config())
sys.exit(0 if date.fromisoformat('$D') in set(df.trade_date) else 3)"; then
  echo "$D 不是交易日,跳过"; exit 0
fi
$PY -m ashare_lab daily --date "$D"
$PY -m ashare_lab sectors
$PY -m ashare_lab structure
$PY -m ashare_lab plan-v3 --date "$D"
