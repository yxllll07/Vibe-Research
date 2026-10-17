"""原始数据只增不改的落盘区。

每个原始响应按 <root>/raw/<source>/<yyyy>/<key><ext> 保存,同时向 raw/manifest.jsonl 追加一行
{source, key, url, sha256, bytes, fetched_at, http_status}。已存在的文件不覆盖:续传时直接复用,
并核对它的 sha256 与清单一致;不一致说明文件被改过,直接报错。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import TZ


class RawIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawRecord:
    source: str
    key: str
    path: Path
    sha256: str
    url: str
    fetched_at: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RawStore:
    def __init__(self, root: Path) -> None:
        self.root = root / "raw"
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "manifest.jsonl"
        self._index: dict[tuple[str, str], dict[str, Any]] | None = None

    def _load_index(self) -> dict[tuple[str, str], dict[str, Any]]:
        if self._index is None:
            self._index = {}
            if self.manifest.exists():
                for line in self.manifest.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        self._index[(row["source"], row["key"])] = row
        return self._index

    def path_for(self, source: str, key: str, ext: str) -> Path:
        return self.root / source / key[:4] / f"{key}{ext}"

    def get(self, source: str, key: str) -> RawRecord | None:
        row = self._load_index().get((source, key))
        if row is None:
            return None
        path = self.root / row["relpath"]
        if not path.exists():
            raise RawIntegrityError(f"清单里有 {source}/{key},文件却不见了:{path}")
        digest = sha256_bytes(path.read_bytes())
        if digest != row["sha256"]:
            raise RawIntegrityError(f"{path} 的 sha256 与清单不符,原始文件被改动过")
        return RawRecord(source, key, path, digest, row["url"], row["fetched_at"])

    def put(self, source: str, key: str, ext: str, data: bytes, url: str, http_status: int) -> RawRecord:
        existing = self._load_index().get((source, key))
        if existing is not None:
            raise RawIntegrityError(f"{source}/{key} 已落盘,原始区不覆盖;要新版本请换 key")
        path = self.path_for(source, key, ext)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        row: dict[str, Any] = {
            "source": source,
            "key": key,
            "relpath": str(path.relative_to(self.root)),
            "url": url,
            "sha256": sha256_bytes(data),
            "bytes": len(data),
            "fetched_at": datetime.now(TZ).isoformat(timespec="seconds"),
            "http_status": http_status,
        }
        with self.manifest.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._load_index()[(source, key)] = row
        return RawRecord(source, key, path, str(row["sha256"]), url, str(row["fetched_at"]))
