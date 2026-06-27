"""Small JSON-RPC style tool server for agents.

This is not a full MCP implementation; it is a dependency-free local tool
adapter that exposes the same safe operations the MCP server should expose.
If your environment already provides an MCP framework, wrap these handlers in
that framework rather than allowing agents direct DB/API access.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable

from .agents import HeuristicTuningAgent
from .config import RuntimeConfig
from .csv_patch import CsvMaterializer, PatchValidator
from .models import FailureSummary, tuning_candidate_from_json, to_jsonable
from .registry import ExperimentRegistry

JsonDict = dict[str, Any]


class ToolServer:
    def __init__(self, config: RuntimeConfig | None = None):
        self.config = config or RuntimeConfig.from_env()
        self.registry = ExperimentRegistry(self.config.db_path)
        self.tools: dict[str, Callable[[JsonDict], JsonDict]] = {
            "propose_tuning_patch": self.propose_tuning_patch,
            "validate_tuning_patch": self.validate_tuning_patch,
            "materialize_csv": self.materialize_csv,
            "get_candidate": self.get_candidate,
        }

    def handle(self, request: JsonDict) -> JsonDict:
        tool = str(request.get("tool", ""))
        args = dict(request.get("args", {}))
        if tool not in self.tools:
            return {"ok": False, "error": f"unknown tool: {tool}"}
        try:
            return {"ok": True, "result": self.tools[tool](args)}
        except Exception as exc:  # noqa: BLE001 - JSON-RPC surface
            return {"ok": False, "error": str(exc)}

    def propose_tuning_patch(self, args: JsonDict) -> JsonDict:
        failures = [FailureSummary(**item) for item in args.get("failures", [])]
        candidates = HeuristicTuningAgent().propose_candidates(
            failures=failures,
            base_csv_id=str(args["base_csv_id"]),
            row_selector=dict(args["row_selector"]),
            max_candidates=int(args.get("max_candidates", 5)),
            parent_tuning_ids=list(args.get("parent_tuning_ids", [])),
        )
        for candidate in candidates:
            self.registry.add_candidate(candidate)
        return {"candidates": to_jsonable(candidates)}

    def validate_tuning_patch(self, args: JsonDict) -> JsonDict:
        candidate = tuning_candidate_from_json(dict(args["candidate"]))
        report = PatchValidator(max_instruction_chars=self.config.search_policy.max_instruction_chars).validate(
            candidate.patch,
            base_csv_path=args.get("base_csv_path"),
            reference_texts=args.get("reference_texts", []),
            case_specific_values=args.get("case_specific_values", []),
        )
        return {"report": to_jsonable(report)}

    def materialize_csv(self, args: JsonDict) -> JsonDict:
        candidate = tuning_candidate_from_json(dict(args["candidate"]))
        result = CsvMaterializer().materialize(args["base_csv_path"], candidate, args["output_path"])
        return to_jsonable(result)

    def get_candidate(self, args: JsonDict) -> JsonDict:
        candidate = self.registry.get_candidate(str(args["tuning_id"]))
        return {"candidate": to_jsonable(candidate) if candidate else None}


def main() -> int:
    server = ToolServer()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        response = server.handle(json.loads(line))
        print(json.dumps(response, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
