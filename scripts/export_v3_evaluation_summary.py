"""Export the v3 human-reference evaluation summary."""

from __future__ import annotations

import argparse

from poc_automation.registry import ExperimentRegistry
from poc_automation.v3_summary import export_v3_evaluation_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--status-note", default=None)
    args = parser.parse_args()

    out = export_v3_evaluation_summary(
        ExperimentRegistry(args.db),
        args.out,
        dataset_path=args.dataset,
        status_note=args.status_note,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
