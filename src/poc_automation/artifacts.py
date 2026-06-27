"""Artifact storage abstraction.

For the prototype, a content-addressable local filesystem store is sufficient.
The same interface can later be backed by S3-compatible object storage.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import to_jsonable


@dataclass(frozen=True)
class StoredArtifact:
    uri: str
    sha256: str
    size_bytes: int


class LocalArtifactStore:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _bucket_dir(self, bucket: str) -> Path:
        path = self.root_dir / bucket
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_bytes(self, bucket: str, name: str, payload: bytes) -> StoredArtifact:
        digest = hashlib.sha256(payload).hexdigest()
        suffix = Path(name).suffix
        filename = f"{Path(name).stem}.{digest[:12]}{suffix}"
        path = self._bucket_dir(bucket) / filename
        path.write_bytes(payload)
        return StoredArtifact(uri=str(path), sha256=f"sha256:{digest}", size_bytes=len(payload))

    def write_text(self, bucket: str, name: str, text: str) -> StoredArtifact:
        return self.write_bytes(bucket, name, text.encode("utf-8"))

    def write_json(self, bucket: str, name: str, payload: Any) -> StoredArtifact:
        text = json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        return self.write_text(bucket, name, text + "\n")

    def copy_file(self, bucket: str, path: str | Path) -> StoredArtifact:
        source = Path(path)
        payload = source.read_bytes()
        stored = self.write_bytes(bucket, source.name, payload)
        return stored

    def read_json(self, uri: str) -> Any:
        return json.loads(Path(uri).read_text(encoding="utf-8"))

    def mirror_tree(self, bucket: str, source_dir: str | Path) -> Path:
        source = Path(source_dir)
        target = self._bucket_dir(bucket) / source.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        return target
