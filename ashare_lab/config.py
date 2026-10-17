"""配置加载与数据根目录。"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "eod_v1.yaml"
TZ = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def data_root(self) -> Path:
        env = os.environ.get("ASHARE_LAB_DATA")
        if env:
            return Path(env)
        configured = self.raw.get("data", {}).get("root")
        return Path(configured) if configured else REPO_ROOT / ".local" / "ashare_lab"

    def hash(self) -> str:
        """配置的稳定哈希;锁定模型前记录,盲测揭盲后不得再改配置。"""
        blob = json.dumps(self.raw, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG
    with p.open(encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh), p)
