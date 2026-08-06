"""T3-03: CTRAG adaptive chunking A/B experiment (NR-03).

Compares retrieval quality between the existing fixed-size splitter
(CharacterTextSplitter 1500/200) and the adaptive chunker on real
regulatory documents, using the deterministic claim-level metrics.

Reference maturity: CTRAG reports a POC at one Big-4 firm, not production
deployment; gains are reference signals only.

Usage: python -m app.services.retrieval.ctrag_ab_experiment
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.services.evaluation.benchmark_loader import load_benchmark_questions
from app.services.evaluation.rag_eval import evaluate_claim_level_metrics
from app.services.retrieval.adaptive_chunker import adaptive_chunk_text


def _load_source_texts() -> list[str]:
    """Load raw regulatory document texts (pre-chunking source)."""
    import json

    cache = json.load(
        open(Path(__file__).resolve().parents[3] / "data" / "indexes" / "corpus_documents.json", encoding="utf-8")
    )
    docs = cache.get("documents", [])
    # Group chunks by doc_id and reconstruct a "raw-ish" document.
    by_doc: dict[str, list[str]] = {}
    for doc in docs:
        doc_id = str(doc.get("metadata", {}).get("doc_id", "unknown"))
        by_doc.setdefault(doc_id, []).append(str(doc.get("page_content", "")))
    return ["\n".join(chunks) for chunks in by_doc.values() if chunks]


def _fixed_split(text: str, chunk_size: int = 1500, overlap: int = 200) -> list:
    """Existing fixed-size splitter (matches builder.py fallback semantics).

    CharacterTextSplitter splits on paragraph breaks by default; the
    reconstructed corpus text has dense single newlines, so force a
    separator-aware config that actually produces multiple chunks.
    """
    from langchain_core.documents import Document
    from langchain_text_splitters import CharacterTextSplitter

    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separator="\n",
    )
    return [Document(page_content=chunk, metadata={}) for chunk in splitter.split_text(text)]


def _evidence_pool(chunks: list) -> list[dict]:
    """Deterministic BM25-ish overlap retrieval over a chunk pool."""
    from app.services.evaluation.run_eval import _tokens

    pool = []
    for i, chunk in enumerate(chunks):
        text = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
        pool.append((i, text, _tokens(text)))
    return pool


def _retrieve(pool: list, query: str, top_k: int = 6) -> list:
    from app.services.evaluation.run_eval import _tokens

    qt = _tokens(query)
    scored = []
    for i, text, tokens in pool:
        overlap = len(qt & tokens)
        if overlap > 0:
            scored.append((overlap, i, text))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [{"page_content": text} for _, _, text in scored[:top_k]]


def run_ab() -> dict:
    questions = load_benchmark_questions()
    sources = _load_source_texts()
    print(f"documents reconstructed: {len(sources)}")

    fixed_pool = []
    adaptive_pool = []
    for source in sources:
        fixed_pool.extend(_evidence_pool(_fixed_split(source)))
        adaptive_pool.extend(_evidence_pool(adaptive_chunk_text(source)))
    print(f"fixed chunks: {len(fixed_pool)}, adaptive chunks: {len(adaptive_pool)}")

    rows = []
    for item in questions[:60]:
        claims = item.get("expected_claims", [])
        fixed_evidence = _retrieve(fixed_pool, item["question"])
        adaptive_evidence = _retrieve(adaptive_pool, item["question"])
        fixed_metrics = evaluate_claim_level_metrics(claims, fixed_evidence)
        adaptive_metrics = evaluate_claim_level_metrics(claims, adaptive_evidence)
        rows.append(
            {
                "id": item["id"],
                "fixed_claim_recall": fixed_metrics["claim_recall"],
                "adaptive_claim_recall": adaptive_metrics["claim_recall"],
                "fixed_context_precision": fixed_metrics["context_precision"],
                "adaptive_context_precision": adaptive_metrics["context_precision"],
            }
        )

    n = len(rows)
    avg = lambda key: sum(r[key] for r in rows) / n if n else 0.0
    summary = {
        "scenarios": n,
        "fixed_claim_recall": round(avg("fixed_claim_recall"), 3),
        "adaptive_claim_recall": round(avg("adaptive_claim_recall"), 3),
        "fixed_context_precision": round(avg("fixed_context_precision"), 3),
        "adaptive_context_precision": round(avg("adaptive_context_precision"), 3),
        "adaptive_wins": sum(
            1 for r in rows if r["adaptive_claim_recall"] > r["fixed_claim_recall"]
        ),
        "fixed_wins": sum(
            1 for r in rows if r["adaptive_claim_recall"] < r["fixed_claim_recall"]
        ),
        "ties": sum(
            1 for r in rows if r["adaptive_claim_recall"] == r["fixed_claim_recall"]
        ),
    }
    return summary


def main() -> None:
    summary = run_ab()
    print("=" * 60)
    print("CTRAG adaptive vs fixed chunking A/B")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
