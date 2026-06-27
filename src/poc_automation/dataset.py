"""Dataset loading and snapshotting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import Case, Split, case_from_json, to_jsonable


@dataclass(frozen=True)
class Dataset:
    dataset_id: str
    snapshot_id: str
    cases: list[Case]
    metadata: dict[str, object]

    def by_split(self, split: Split | str) -> list[Case]:
        split_value = split.value if isinstance(split, Split) else split
        return [case for case in self.cases if case.split.value == split_value]

    def select(self, splits: Iterable[Split | str]) -> list[Case]:
        split_values = {split.value if isinstance(split, Split) else split for split in splits}
        return [case for case in self.cases if case.split.value in split_values]


def _hash_manifest_payload(payload: object) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return "ds_" + hashlib.sha256(data).hexdigest()[:16]


def load_dataset_manifest(path: str | Path) -> Dataset:
    manifest_path = Path(path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = str(manifest_path.parent)
    cases = [case_from_json(item, base_dir=base_dir) for item in raw.get("cases", [])]
    payload_for_hash = {
        "dataset_id": raw.get("dataset_id"),
        "cases": raw.get("cases", []),
        "metadata": raw.get("metadata", {}),
    }
    snapshot_id = raw.get("snapshot_id") or _hash_manifest_payload(payload_for_hash)
    return Dataset(
        dataset_id=str(raw.get("dataset_id", manifest_path.stem)),
        snapshot_id=str(snapshot_id),
        cases=cases,
        metadata=dict(raw.get("metadata", {})),
    )


def dataset_to_langfuse_local_data(dataset: Dataset) -> list[dict[str, object]]:
    """Return a Langfuse experiment-compatible local dataset shape."""

    items: list[dict[str, object]] = []
    for case in dataset.cases:
        items.append(
            {
                "input": {
                    "case_id": case.case_id,
                    "procedure_csv_path": case.procedure_csv_path,
                    "evidence_bundle_path": case.evidence_bundle_path,
                    "metadata": case.metadata,
                },
                "expected_output": to_jsonable(case.expected_output),
                "metadata": {
                    "split": case.split.value,
                    "dataset_snapshot_id": dataset.snapshot_id,
                },
            }
        )
    return items
