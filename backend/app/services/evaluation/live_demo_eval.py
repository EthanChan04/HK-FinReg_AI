"""Capture grounded responses from the explicit DeepSeek demo runtime."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.services.evaluation.benchmark_loader import load_benchmark_questions
from app.services.evaluation.run_eval import ResponseProvider, retrieve_eval_documents
from app.services.llm.deepseek import build_deepseek_llm


LIVE_DEMO_CASE_IDS = (
    "RAG_SVF_AML_001",
    "KAG_AI_ADVISOR_001",
    "DR_AI_ADVISOR_001",
    "EXP_051",
    "EXP_062",
    "EXP_071",
    "EXP_077",
    "EXP_082",
    "EXP_088",
    "EXP_089",
    "EXP_090",
    "EXP_095",
)
SCHEMA_VERSION = 1
PROVIDER = "deepseek"
MODEL = "deepseek-v4-flash"
PROMPT_VERSION = "demo-grounded-v1"
ARTIFACT_FILENAME = "deepseek-v4-flash-live-responses.json"


def _evidence_id(document: Document, index: int) -> str:
    metadata = document.metadata or {}
    base = (
        metadata.get("chunk_id")
        or metadata.get("doc_id")
        or metadata.get("source_id")
        or metadata.get("source")
        or f"evidence-{index}"
    )
    page = metadata.get("page") or metadata.get("page_number")
    return f"{base}#page={page}" if page is not None else str(base)


def build_grounded_prompt(item: dict, evidence: list[Document]) -> list:
    """Build a versioned, evidence-only prompt with numbered citations."""

    rendered_evidence = []
    for index, document in enumerate(evidence, start=1):
        rendered_evidence.append(
            f"[{index}] {_evidence_id(document, index)}\n{document.page_content.strip()}"
        )
    evidence_text = "\n\n".join(rendered_evidence) or "(no evidence retrieved)"
    return [
        SystemMessage(
            content=(
                "You are a Hong Kong financial-regulation compliance assistant. "
                "Answer only from the supplied evidence. Cite supporting evidence "
                "with bracketed numbers such as [1]. If the evidence is insufficient, "
                "say explicitly that the evidence is insufficient and identify the gap. "
                "Do not invent obligations, dates, regulators, or sources."
            )
        ),
        HumanMessage(
            content=(
                f"Case ID: {item.get('id', '')}\n"
                f"Question: {item.get('question', '')}\n\n"
                f"Evidence:\n{evidence_text}"
            )
        ),
    ]


def _status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _is_retryable(exc: Exception) -> bool:
    status = _status_code(exc)
    return status == 429 or (status is not None and 500 <= status <= 599)


def _is_auth_failure(exc: Exception) -> bool:
    return _status_code(exc) in {401, 403}


def _safe_error(exc: Exception) -> str:
    """Return useful error metadata without persisting credentials."""

    status = _status_code(exc)
    message = str(exc)
    message = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [REDACTED]", message)
    message = re.sub(r"(?i)\bsk-[A-Za-z0-9_-]+", "[REDACTED]", message)
    message = re.sub(
        r"(?i)(api[_ -]?key\s*[=:]\s*)[^\s,;]+",
        r"\1[REDACTED]",
        message,
    )
    prefix = f"HTTP {status}: " if status is not None else f"{type(exc).__name__}: "
    return (prefix + message)[:500]


def _response_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if not isinstance(content, str):
        raise ValueError("malformed model response: content must be text")
    return content.strip()


def _usage(response: Any) -> dict:
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return {str(key): value for key, value in usage.items()}
    metadata = getattr(response, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage") if isinstance(metadata, dict) else None
    return dict(token_usage) if isinstance(token_usage, dict) else {}


def _selected_items() -> list[dict]:
    by_id = {str(item.get("id")): item for item in load_benchmark_questions()}
    missing = [case_id for case_id in LIVE_DEMO_CASE_IDS if case_id not in by_id]
    if missing:
        raise ValueError(f"live demo benchmark cases are missing: {', '.join(missing)}")
    return [by_id[case_id] for case_id in LIVE_DEMO_CASE_IDS]


def capture_live_responses(output_dir: str | Path, llm=None) -> dict:
    """Call the real DeepSeek runtime and persist a redacted response artifact."""

    runtime = llm or build_deepseek_llm("evaluation")
    cases = []
    fatal_reason: str | None = None
    for item in _selected_items():
        case_id = str(item["id"])
        if fatal_reason:
            cases.append(
                {
                    "case_id": case_id,
                    "response": "",
                    "evidence_ids": [],
                    "latency_ms": 0,
                    "usage": {},
                    "error": fatal_reason,
                }
            )
            continue

        evidence = retrieve_eval_documents(str(item["question"]), top_k=10)
        evidence_ids = [_evidence_id(document, index) for index, document in enumerate(evidence, 1)]
        started = time.perf_counter()
        model_response = None
        error: str | None = None
        for attempt in range(3):
            try:
                model_response = runtime.invoke(build_grounded_prompt(item, evidence))
                break
            except Exception as exc:
                error = _safe_error(exc)
                if _is_auth_failure(exc):
                    fatal_reason = "not attempted after fatal authentication failure"
                    break
                if not _is_retryable(exc) or attempt == 2:
                    break
                time.sleep(attempt + 1)

        response_text = ""
        usage = {}
        if model_response is not None:
            try:
                response_text = _response_text(model_response)
                usage = _usage(model_response)
                error = None
                if not response_text:
                    error = "empty model response"
                    fatal_reason = "not attempted after fatal empty model response"
            except ValueError as exc:
                error = str(exc)
                fatal_reason = "not attempted after fatal malformed model response"
        cases.append(
            {
                "case_id": case_id,
                "response": response_text,
                "evidence_ids": evidence_ids,
                "latency_ms": max(0, round((time.perf_counter() - started) * 1000)),
                "usage": usage,
                "error": error,
            }
        )

    document = {
        "schema_version": SCHEMA_VERSION,
        "provider": PROVIDER,
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "cases": cases,
    }
    destination = Path(output_dir) / ARTIFACT_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return document


def load_live_response_provider(document: dict) -> ResponseProvider:
    """Validate a live artifact and adapt it to the existing evaluation API."""

    if not isinstance(document, dict):
        raise ValueError("live response artifact must be a JSON object")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported live response schema_version")
    if document.get("provider") != PROVIDER:
        raise ValueError("live response provider must be deepseek")
    if document.get("model") != MODEL:
        raise ValueError(f"live response model must be {MODEL}")
    if document.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("unsupported live response prompt_version")
    cases = document.get("cases")
    if not isinstance(cases, list):
        raise ValueError("live response cases must be a list")
    responses: dict[str, str] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each live response case must be an object")
        case_id = case.get("case_id")
        response = case.get("response")
        if not isinstance(case_id, str) or not isinstance(response, str):
            raise ValueError("each live response case requires text case_id and response")
        if case_id in responses:
            raise ValueError(f"duplicate live response case: {case_id}")
        responses[case_id] = response

    def provide(item: dict, evidence: list[Document]) -> str | None:
        del evidence
        return responses.get(str(item.get("id", "")))

    return provide


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evaluation/live"),
    )
    args = parser.parse_args(argv)
    document = capture_live_responses(args.output_dir)
    successes = sum(1 for case in document["cases"] if not case["error"])
    print(f"Captured {successes}/{len(document['cases'])} DeepSeek responses")
    if successes != len(document["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
