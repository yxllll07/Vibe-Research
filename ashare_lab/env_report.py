"""运行环境扫描:Python、CPU、内存、磁盘、依赖、网络可达性;不假设 GPU / AVX2 / Tushare 权限。"""
from __future__ import annotations

import importlib.metadata as md
import os
import platform
import shutil
from datetime import datetime
from pathlib import Path

import requests

from .config import PACKAGE_ROOT, TZ, Config

DEPS = ["pandas", "numpy", "pyarrow", "duckdb", "pyyaml", "scikit-learn", "lightgbm", "torch", "akshare", "requests"]
HOSTS = {
    "通达信盘后包": "https://www.tdx.com.cn/products/data/data/g4day/",
    "新浪交易日历": "https://finance.sina.com.cn/",
    "东财涨停池": "https://push2ex.eastmoney.com/",
    "Tushare": "https://api.tushare.pro/",
}


def _cpu_flags() -> set[str]:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def _mem_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    return None


def write_report(cfg: Config) -> Path:
    lines = [f"# ENVIRONMENT_REPORT", "", f"生成时间:{datetime.now(TZ).isoformat(timespec='seconds')}", ""]
    flags = _cpu_flags()
    disk = shutil.disk_usage(cfg.data_root if cfg.data_root.exists() else Path.cwd())
    lines += ["## 机器", "",
              f"- Python {platform.python_version()} · {platform.system()} {platform.machine()}",
              f"- CPU 核数 {os.cpu_count()} · AVX2 {'有' if 'avx2' in flags else '无/未知'}",
              f"- 内存 {_mem_gb()} GB · 数据目录可用磁盘 {disk.free / 1e9:.1f} GB",
              f"- GPU:不检测、不依赖(M0/M1 只用 CPU)", "",
              "## 依赖", "", "| 包 | 版本 |", "|---|---|"]
    for dep in DEPS:
        try:
            v = md.version(dep)
        except md.PackageNotFoundError:
            v = "未安装"
        lines.append(f"| {dep} | {v} |")
    lines += ["", "## 网络", "", "| 数据源 | 结果 |", "|---|---|"]
    for name, url in HOSTS.items():
        try:
            r = requests.get(url, timeout=10)
            res = f"可达(HTTP {r.status_code})"
        except Exception as exc:
            res = f"不可达:{type(exc).__name__}"
        lines.append(f"| {name} | {res} |")
    lines += ["", "## 凭据", "", f"- TUSHARE_TOKEN:{'已设置' if os.environ.get('TUSHARE_TOKEN') else '未设置(v0.1 不需要)'}",
              "- baostock:走自有 TCP 协议,云端代理环境下不可用,v0.1 不依赖", ""]
    out = PACKAGE_ROOT / "ENVIRONMENT_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
