"""PoC application runners.

`HttpPocAppRunner` wraps the real upload/execute/fetch API.
`DeepAgentPocAppRunner` is a LangChain DeepAgents-based target agent backed by
OpenRouter/Qwen. `MockPocAppRunner` keeps the repository executable without
external services and is used by tests and examples.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import AppApiConfig, TargetAgentConfig
from .evidence import read_text_artifact
from .models import AppRunResult, Case, Citation, NormalizedResult, RationaleItem

JsonDict = dict[str, Any]


class PocAppRunner(Protocol):
    def run_case(self, *, case: Case, materialized_csv_path: str) -> AppRunResult: ...


@dataclass(frozen=True)
class HttpEndpointMap:
    upload_evidence: str = "/evidence"
    upload_csv: str = "/procedure-csv"
    execute: str = "/runs"
    fetch_result_template: str = "/runs/{run_id}"


class HttpPocAppRunner:
    """Thin adapter for the real PoC application API."""

    def __init__(self, config: AppApiConfig, endpoints: HttpEndpointMap | None = None):
        self.config = config
        self.endpoints = endpoints or HttpEndpointMap()

    def run_case(self, *, case: Case, materialized_csv_path: str) -> AppRunResult:
        started = time.monotonic()
        uploaded_evidence = self._post_json(
            self.endpoints.upload_evidence,
            {"case_id": case.case_id, "evidence_bundle_path": case.evidence_bundle_path},
        )
        uploaded_csv = self._post_json(
            self.endpoints.upload_csv,
            {"case_id": case.case_id, "csv_path": materialized_csv_path},
        )
        run = self._post_json(
            self.endpoints.execute,
            {
                "case_id": case.case_id,
                "uploaded_evidence_id": uploaded_evidence.get("id"),
                "uploaded_csv_id": uploaded_csv.get("id"),
            },
        )
        app_run_id = str(run.get("run_id") or run.get("id"))
        raw = self._poll_result(app_run_id)
        normalized = normalize_app_response(raw)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        return AppRunResult(
            app_run_id=app_run_id,
            normalized_result=normalized,
            raw_output=raw,
            latency_ms=elapsed_ms,
            cost=raw.get("cost", {}) if isinstance(raw, dict) else {},
        )

    def _poll_result(self, run_id: str) -> dict[str, object]:
        path = self.endpoints.fetch_result_template.format(run_id=urllib.parse.quote(run_id))
        last: dict[str, object] = {}
        for _ in range(self.config.max_polls):
            last = self._get_json(path)
            status = str(last.get("status", "succeeded"))
            if status in {"succeeded", "failed", "error"}:
                return last
            time.sleep(self.config.poll_interval_seconds)
        raise TimeoutError(f"PoC app run did not finish: {run_id}; last={last}")

    def _url(self, path: str) -> str:
        return urllib.parse.urljoin(self.config.base_url.rstrip("/") + "/", path.lstrip("/"))

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self._url(path), data=data, headers=self._headers(), method="POST")
        return self._request_json(request)

    def _get_json(self, path: str) -> dict[str, object]:
        request = urllib.request.Request(self._url(path), headers=self._headers(), method="GET")
        return self._request_json(request)

    def _request_json(self, request: urllib.request.Request) -> dict[str, object]:
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:  # noqa: S310 - configured endpoint
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"PoC API HTTP {exc.code}: {body}") from exc


class MockPocAppRunner:
    """Deterministic local runner that simulates the target AI app."""

    def run_case(self, *, case: Case, materialized_csv_path: str) -> AppRunResult:
        started = time.monotonic()
        instruction_text = _read_all_instructions(materialized_csv_path)
        expected = case.expected_output
        grounding = any(token in instruction_text for token in ["証跡", "引用", "明示", "推測"])
        condition_check = any(token in instruction_text for token in ["条件", "分岐", "必須", "任意"])
        abstention = any(token in instruction_text for token in ["判断不能", "不足"])

        # Simulate baseline errors for cases that need specific capabilities.
        required_capability = str(case.metadata.get("required_capability", "evidence_grounding"))
        capability_ok = {
            "evidence_grounding": grounding,
            "citation_precision": grounding and "引用" in instruction_text,
            "condition_check": condition_check,
            "abstention": abstention,
        }.get(required_capability, grounding or condition_check or abstention)

        judgement = expected.judgement if capability_ok else _wrong_judgement(expected.judgement)
        citations = expected.citations if grounding else []
        claim = _make_claim(expected.required_claim_keywords, capability_ok=capability_ok)
        if not grounding and capability_ok:
            claim += "。なお、証跡外の推測を含む可能性があります"

        normalized = NormalizedResult(
            judgement=judgement,
            rationale_items=[RationaleItem(claim=claim, citations=citations)],
            raw_output={"mock": True, "capability_ok": capability_ok, "instruction_text": instruction_text},
            warnings=[] if capability_ok else [f"missing_capability:{required_capability}"],
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        return AppRunResult(
            app_run_id=f"mock_{uuid.uuid4().hex[:10]}",
            normalized_result=normalized,
            raw_output=normalized.raw_output,
            latency_ms=elapsed_ms,
            cost={"estimated_tokens": len(instruction_text) // 3 + 50},
        )


class DeepAgentPocAppRunner:
    """Evaluation target built with LangChain Deep Agents and OpenRouter/Qwen.

    This runner gives the search loop a local LLM-based target application when
    the external PoC API is not yet available. It creates a Deep Agent with
    read-only tools for the materialized CSV and evidence bundle.
    """

    def __init__(self, config: TargetAgentConfig | None = None):
        self.config = config or TargetAgentConfig.from_env()

    def run_case(self, *, case: Case, materialized_csv_path: str) -> AppRunResult:
        started = time.monotonic()
        app_run_id = f"deepagent_{uuid.uuid4().hex[:10]}"
        try:
            result = self._invoke_agent(case=case, materialized_csv_path=materialized_csv_path)
            output_text = _extract_agent_text(result)
            raw = parse_deepagent_json_response(output_text)
            normalized = normalize_deepagent_response(raw)
            elapsed_ms = int((time.monotonic() - started) * 1000)
            cost = _extract_usage_metadata(result) or {"estimated_tokens": max(1, len(output_text) // 3)}
            return AppRunResult(
                app_run_id=app_run_id,
                normalized_result=normalized,
                raw_output={
                    "deepagent": True,
                    "provider": self.config.provider,
                    "model": self.config.model,
                    "raw_agent_text": output_text,
                    "parsed": raw,
                },
                latency_ms=elapsed_ms,
                cost=cost,
            )
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            return AppRunResult(
                app_run_id=app_run_id,
                normalized_result=NormalizedResult(
                    judgement="",
                    rationale_items=[],
                    raw_output={"error": str(exc)},
                    warnings=["deepagent_runner_failed"],
                ),
                raw_output={"deepagent": True, "model": self.config.model, "error": str(exc)},
                latency_ms=elapsed_ms,
                cost={},
                status="failed",
                error_message=str(exc),
            )

    def _invoke_agent(self, *, case: Case, materialized_csv_path: str) -> Any:
        self._prepare_openrouter_env()
        if not self.config.use_deepagent_tools:
            return self._invoke_openrouter_http(case=case, materialized_csv_path=materialized_csv_path)

        try:
            from deepagents import create_deep_agent  # type: ignore
            from langchain_openrouter import ChatOpenRouter  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "DeepAgentPocAppRunner requires optional dependencies. "
                "Install with: pip install -e .[target-agent]"
            ) from exc

        model_kwargs: dict[str, object] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "max_retries": self.config.max_retries,
            "timeout": self.config.timeout_seconds,
        }
        if self.config.openrouter_provider:
            model_kwargs["openrouter_provider"] = self.config.openrouter_provider
        if self.config.route:
            model_kwargs["route"] = self.config.route
        if self.config.app_url:
            model_kwargs["app_url"] = self.config.app_url
        if self.config.app_title:
            model_kwargs["app_title"] = self.config.app_title

        model = ChatOpenRouter(**model_kwargs)
        def read_procedure_csv() -> str:
            """Read the full materialized procedure CSV for the current case."""
            return read_text_artifact(materialized_csv_path, max_chars=self.config.max_csv_chars)

        def read_evidence_bundle() -> str:
            """Read the evidence bundle for the current case as text/JSON."""
            return read_text_artifact(case.evidence_bundle_path, max_chars=self.config.max_evidence_chars)

        agent = create_deep_agent(
            model=model,
            tools=[read_procedure_csv, read_evidence_bundle],
            system_prompt=self.config.system_prompt,
        )
        prompt = build_deepagent_prompt(case)
        return agent.invoke({"messages": [{"role": "user", "content": prompt}]})

    def _invoke_openrouter_http(self, *, case: Case, materialized_csv_path: str) -> dict[str, object]:
        prompt = build_target_agent_prompt(case=case, materialized_csv_path=materialized_csv_path)
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.openrouter_provider:
            payload["provider"] = self.config.openrouter_provider
        if self.config.route:
            payload["route"] = self.config.route

        api_key = self.config.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY or POC_TARGET_AGENT_API_KEY is required for --runner deepagent")
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        app_url = self.config.app_url or os.getenv("OPENROUTER_HTTP_REFERER") or os.getenv("OPENROUTER_APP_URL")
        if app_url:
            headers["HTTP-Referer"] = app_url
        app_title = self.config.app_title or os.getenv("OPENROUTER_APP_TITLE")
        if app_title:
            headers["X-Title"] = app_title

        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:  # noqa: S310
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body[:1000]}") from exc

        choices = raw.get("choices") if isinstance(raw, dict) else None
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"OpenRouter response did not include choices: {str(raw)[:500]}")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"OpenRouter response did not include message content: {str(raw)[:500]}")

        raw_usage = raw.get("usage") if isinstance(raw, dict) else None
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        usage_metadata = {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        return {
            "messages": [{"content": content, "usage_metadata": usage_metadata}],
            "openrouter_http": True,
            "raw_response": raw,
        }

    def _prepare_openrouter_env(self) -> None:
        if self.config.provider != "openrouter":
            raise ValueError(f"Unsupported target agent provider: {self.config.provider}")
        if self.config.api_key:
            os.environ.setdefault("OPENROUTER_API_KEY", self.config.api_key)
        if self.config.base_url:
            os.environ.setdefault("OPENROUTER_BASE_URL", self.config.base_url)
        if not os.getenv("OPENROUTER_API_KEY"):
            raise RuntimeError("OPENROUTER_API_KEY or POC_TARGET_AGENT_API_KEY is required for --runner deepagent")
        if self.config.app_url:
            os.environ.setdefault("OPENROUTER_HTTP_REFERER", self.config.app_url)
            os.environ.setdefault("OPENROUTER_APP_URL", self.config.app_url)
        if self.config.app_title:
            os.environ.setdefault("OPENROUTER_APP_TITLE", self.config.app_title)
            os.environ.setdefault("OPENROUTER_X_TITLE", self.config.app_title)


DEEPAGENT_TARGET_SYSTEM_PROMPT = """あなたは業務手続の評価エージェントです。
手続CSVと証跡だけを根拠に、評価結果、根拠、引用を生成します。

厳守事項:
- 必ず手続CSVと証跡を確認してから判断する。
- 必要に応じて read_procedure_csv と read_evidence_bundle を使う。
- 人手正解や期待値は存在しないものとして扱う。
- 証跡に明示されていない推測、一般知識、補完を根拠にしない。
- 根拠ごとに、その根拠を直接支持する引用を付ける。
- 引用できない主張は根拠として出力しない。
- 証跡不足で判定できない場合は judgement を「判断不能」にする。
- 出力はJSONのみ。Markdown、コードフェンス、説明文は出力しない。
"""


def build_deepagent_prompt(case: Case) -> str:
    """Build a tool-oriented target-agent prompt without inline artifacts."""

    return f"""次のケースを評価してください。

case_id: {case.case_id}

手順:
1. read_procedure_csv で手続CSVを読む。
2. read_evidence_bundle で証跡を読む。
3. 手続CSVの条件と追加指示に従って、証跡だけを根拠に判断する。
4. 次のJSONスキーマだけを出力する。

{_target_output_schema()}
"""


def build_target_agent_prompt_payload(*, case: Case, materialized_csv_path: str | Path) -> dict[str, object]:
    """Return a DeepAgent target prompt payload without human reference data."""

    return {
        "case_id": case.case_id,
        "split": case.split.value,
        "metadata": _safe_case_metadata(case),
        "procedure_csv": read_text_artifact(str(materialized_csv_path), max_chars=12000),
        "evidence_bundle": read_text_artifact(case.evidence_bundle_path, max_chars=30000),
        "output_schema": _target_output_schema(),
    }


def build_target_agent_prompt(*, case: Case, materialized_csv_path: str) -> str:
    """Build the prompt for the DeepAgent target without leaking expected output."""

    payload = build_target_agent_prompt_payload(case=case, materialized_csv_path=materialized_csv_path)
    metadata = json.dumps(payload["metadata"], ensure_ascii=False, sort_keys=True)
    procedure_csv = str(payload["procedure_csv"])
    evidence = str(payload["evidence_bundle"])
    return f"""次のケースを評価してください。

case_id: {case.case_id}
metadata: {metadata}

# 手続CSV
{procedure_csv}

# 証跡
{evidence}

手順:
1. 手続CSVの条件と追加指示を確認する。
2. 証跡に明示された内容だけを根拠にする。
3. 判断に必要な情報が不足する場合は「判断不能」とする。
4. 次のJSONスキーマだけを出力する。

{_target_output_schema()}
"""


def _safe_case_metadata(case: Case) -> dict[str, object]:
    blocked = {
        "expected_output",
        "human_reference",
        "reference_answer",
        "required_claim_keywords",
        "citations",
        "required_capability",
    }
    return {key: value for key, value in case.metadata.items() if key not in blocked}


def _target_output_schema() -> str:
    return """JSONスキーマ:
{
  "result": {
    "judgement": "適合 | 不適合 | 判断不能",
    "rationale_items": [
      {
        "claim": "根拠文。証跡に明示された内容のみ。",
        "citations": [
          {
            "evidence_id": "証跡ID",
            "page": 1,
            "span": "根拠文を直接支持する証跡内の短い引用または該当箇所",
            "claim": "この引用が支持する内容"
          }
        ]
      }
    ],
    "warnings": []
  }
}
"""


def normalize_deepagent_response(raw: dict[str, object]) -> NormalizedResult:
    """Normalize a DeepAgent response that may omit the outer result envelope."""

    return normalize_app_response(raw if "result" in raw else {"result": raw})


def normalize_app_response(raw: dict[str, object]) -> NormalizedResult:
    result = raw.get("result", raw)
    if not isinstance(result, dict):
        return NormalizedResult(judgement="", rationale_items=[], raw_output=raw, warnings=["unexpected_response_shape"])

    judgement = str(result.get("judgement") or result.get("evaluation_result") or "")
    rationale_items: list[RationaleItem] = []
    raw_items = result.get("rationale_items") or result.get("rationales") or []
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            citations: list[Citation] = []
            for citation in item.get("citations", []) or []:
                if isinstance(citation, dict):
                    citations.append(
                        Citation(
                            evidence_id=str(citation.get("evidence_id") or citation.get("document_id") or ""),
                            page=_to_optional_int(citation.get("page")),
                            span=citation.get("span") or citation.get("quote"),
                            claim=citation.get("claim"),
                        )
                    )
            claim = str(item.get("claim") or item.get("text") or "")
            if claim:
                rationale_items.append(RationaleItem(claim=claim, citations=citations))
    else:
        text = str(result.get("rationale") or "")
        if text:
            rationale_items.append(RationaleItem(claim=text, citations=[]))

    warnings = result.get("warnings", [])
    return NormalizedResult(
        judgement=judgement,
        rationale_items=rationale_items,
        raw_output=raw,
        warnings=list(warnings) if isinstance(warnings, list) else [],
    )


def parse_deepagent_json_response(text: str) -> dict[str, object]:
    """Parse a DeepAgent final JSON response with markdown-fence tolerance."""

    stripped = _strip_code_fence(text.strip())
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
        if fenced:
            value = json.loads(fenced.group(1))
        else:
            decoder = json.JSONDecoder()
            for index, char in enumerate(stripped):
                if char not in "{[":
                    continue
                try:
                    value, _ = decoder.raw_decode(stripped[index:])
                    break
                except json.JSONDecodeError:
                    continue
            else:
                raise RuntimeError(f"DeepAgent did not return a JSON object: {text[:500]}") from None
    if isinstance(value, list) and value and isinstance(value[0], dict):
        value = value[0]
    if not isinstance(value, dict):
        raise RuntimeError(f"DeepAgent JSON response must be an object: {type(value).__name__}")
    return value


parse_agent_json = parse_deepagent_json_response
parse_agent_json_response = parse_deepagent_json_response
extract_json_object = parse_deepagent_json_response


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_agent_text(result: object) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                text = _message_content(message)
                if text.strip():
                    return text
        for key in ("content", "output", "text"):
            value = result.get(key)
            if isinstance(value, str):
                return value
    text = _message_content(result)
    if text.strip():
        return text
    raise RuntimeError("DeepAgent response did not include final message content")


def _message_content(message: object) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if value is not None:
                    parts.append(str(value))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _extract_usage_metadata(result: object) -> dict[str, object] | None:
    if not isinstance(result, dict):
        return None
    messages = result.get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        metadata = getattr(message, "usage_metadata", None)
        if isinstance(metadata, dict):
            return dict(metadata)
        if isinstance(message, dict) and isinstance(message.get("usage_metadata"), dict):
            return dict(message["usage_metadata"])
    return None


def _read_text_artifact(path: str, *, max_chars: int) -> str:
    artifact_path = Path(path)
    if artifact_path.is_file():
        return _truncate_text(_read_one_text_file(artifact_path), max_chars)
    if not artifact_path.exists():
        return f"パスが存在しません: {path}"

    sections: list[str] = []
    for file_path in sorted(p for p in artifact_path.rglob("*") if p.is_file()):
        if file_path.name.startswith("."):
            continue
        sections.append(f"## {file_path.relative_to(artifact_path)}\n{_read_one_text_file(file_path)}")
    return _truncate_text("\n\n".join(sections), max_chars)


def _read_one_text_file(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return f"{path.name}: binary evidence file is not readable by the local DeepAgent runner"
    if path.suffix.lower() == ".json":
        try:
            return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            return text
    return text


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _to_optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _read_all_instructions(csv_path: str) -> str:
    texts: list[str] = []
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            for key in ["additional_instruction", "instruction", "手続", "手順"]:
                value = row.get(key)
                if value:
                    texts.append(value)
    return "\n".join(texts)


def _wrong_judgement(expected: str) -> str:
    mapping = {"適合": "不適合", "不適合": "適合", "判断不能": "適合"}
    return mapping.get(expected, "判断不能")


def _make_claim(required_keywords: list[str], *, capability_ok: bool) -> str:
    if not required_keywords:
        return "手続条件に基づき判断しました" if capability_ok else "十分な根拠を確認できませんでした"
    if capability_ok:
        return "、".join(required_keywords) + "を確認しました"
    return f"{required_keywords[0]}の確認が不十分です"
