"""T3-01: PEA-CAE A/B experiment on the golden set (NR-01 prototype).

Compares the existing DeepResearch evidence collection against a
PEA-CAE-gated variant using the deterministic evaluation retrieval path
(no LLM calls):

  Control (existing): retrieve top-k evidence per sub-question, then fill
      gaps with a second retrieval round.
  Treatment (PEA-CAE): same first round, but before the gap-filling round
      decide per sub-question whether to escalate to full-text reading of
      the top-ranked source document. Escalation is simulated by reading
      ALL chunks of the best document (full-text proxy).

Outputs an A/B report: quality (claim_recall / context_precision) vs cost
(retrieval calls + full-text chars) per scenario.

Usage: python -m app.services.deepresearch.pea_cae_ab_experiment
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.services.deepresearch.escalation_gate import should_escalate
from app.services.evaluation.benchmark_loader import load_benchmark_questions
from app.services.evaluation.run_eval import _retrieve_eval_documents
from app.services.evaluation.rag_eval import evaluate_claim_level_metrics

SCENARIO_IDS = {"EXP_051", "EXP_052", "EXP_053", "EXP_055", "EXP_056", "EXP_057", "EXP_058", "EXP_059", "EXP_060", "EXP_061", "EXP_062", "EXP_063", "EXP_064", "EXP_065", "EXP_066", "EXP_067", "EXP_068", "EXP_069", "EXP_070", "EXP_071", "EXP_072", "EXP_073", "EXP_074", "EXP_075", "EXP_076", "EXP_077", "EXP_078", "EXP_079", "EXP_083", "EXP_084", "EXP_085", "EXP_086", "EXP_087", "EXP_088", "EXP_089", "EXP_090", "EXP_091", "EXP_092", "EXP_096", "EXP_097", "EXP_098", "EXP_099", "EXP_100", "EXP_101", "EXP_102", "EXP_103", "EXP_104", "EXP_105"}


def _full_text_of_best_document(docs: list) -> str:
    """Simulate full-text reading: concatenate all chunks of the top doc."""
    if not docs:
        return ""
    best = docs[0]
    doc_id = (getattr(best, "metadata", {}) or {}).get("doc_id")
    if not doc_id:
        return str(getattr(best, "page_content", ""))
    from app.core.config import get_settings
    from app.services.corpus.cache import manifest_digest, read_corpus_cache

    backend_root = Path(__file__).resolve().parents[3]
    settings = get_settings()
    cache_path = Path(settings.CORPUS_INDEX_DIR) / "corpus_documents.json"
    manifest_path = backend_root / "data" / "source_manifest.json"
    cached = read_corpus_cache(
        cache_path,
        manifest_digest=manifest_digest(manifest_path),
        parser_version="hierarchy-v1",
    )
    chunks = [
        str(c.page_content)
        for c in cached
        if (c.metadata or {}).get("doc_id") == doc_id
    ]
    return "\n".join(chunks)


def run_scenario(item: dict) -> dict:
    """Run one benchmark case through control and treatment paths."""
    question = item["question"]
    claims = item.get("expected_claims", [])

    # Round 1 (shared): deterministic retrieval
    first_round = _retrieve_eval_documents(question, top_k=6)
    first_metrics = evaluate_claim_level_metrics(claims, first_round)

    # Control: gap-filling round (re-retrieve with same query, wider top_k)
    gap_round = _retrieve_eval_documents(question, top_k=10)
    control_evidence = list({id(d): d for d in first_round + gap_round}.values())
    control_metrics = evaluate_claim_level_metrics(claims, control_evidence)

    # Treatment: PEA-CAE gate before gap-filling
    coverage = first_metrics["claim_recall"] or 0.0
    gap_ratio = 1.0 - coverage
    full_text = _full_text_of_best_document(first_round)
    escalate, diag = should_escalate(
        coverage=coverage,
        gap_ratio=gap_ratio,
        full_text_chars=len(full_text),
    )
    if escalate and full_text:
        # Full-text reading substitutes the second retrieval round.
        treatment_evidence = first_round + [
            type(first_round[0])(
                page_content=full_text,
                metadata={"doc_id": "fulltext", "regulator": "FULLTEXT"},
            )
        ]
    else:
        treatment_evidence = control_evidence
    treatment_metrics = evaluate_claim_level_metrics(claims, treatment_evidence)

    return {
        "id": item["id"],
        "coverage_round1": coverage,
        "escalated": bool(escalate),
        "reason": diag["reason"],
        "full_text_chars": len(full_text),
        "control": {
            "claim_recall": control_metrics["claim_recall"],
            "context_precision": control_metrics["context_precision"],
            "evidence_count": len(control_evidence),
        },
        "treatment": {
            "claim_recall": treatment_metrics["claim_recall"],
            "context_precision": treatment_metrics["context_precision"],
            "evidence_count": len(treatment_evidence),
        },
    }


def main() -> None:
    questions = [q for q in load_benchmark_questions() if q["id"] in SCENARIO_IDS]
    rows = [run_scenario(q) for q in questions]

    escalated = [r for r in rows if r["escalated"]]
    control_cr = sum(r["control"]["claim_recall"] for r in rows) / len(rows)
    treatment_cr = sum(r["treatment"]["claim_recall"] for r in rows) / len(rows)
    control_cp = sum(r["control"]["context_precision"] for r in rows) / len(rows)
    treatment_cp = sum(r["treatment"]["context_precision"] for r in rows) / len(rows)
    control_calls = sum(r["control"]["evidence_count"] for r in rows)
    treatment_calls = sum(r["treatment"]["evidence_count"] for r in rows)

    print("=" * 70)
    print(f"PEA-CAE A/B (scenarios: {len(rows)}, escalated: {len(escalated)})")
    print("=" * 70)
    print(f"{'metric':<28}{'control':>10}{'treatment':>12}")
    print(f"{'claim_recall':<28}{control_cr:>10.3f}{treatment_cr:>12.3f}")
    print(f"{'context_precision':<28}{control_cp:>10.3f}{treatment_cp:>12.3f}")
    print(f"{'evidence units':<28}{control_calls:>10d}{treatment_calls:>12d}")
    print()
    improved = sum(
        1 for r in rows if r["treatment"]["claim_recall"] > r["control"]["claim_recall"]
    )
    round1_covered = sum(1 for r in rows if r["coverage_round1"] >= 0.5)
    print(f"treatment improves claim_recall on {improved}/{len(rows)} scenarios")
    print(f"escalation decisions: {len(escalated)} escalate, {len(rows)-len(escalated)} hold")
    print(f"round-1 coverage >= 0.5 on {round1_covered}/{len(rows)} scenarios")
    print()
    for r in rows:
        if r["escalated"]:
            print(
                f"  {r['id']}: round1={r['coverage_round1']:.2f} -> "
                f"control={r['control']['claim_recall']:.2f} / "
                f"treatment={r['treatment']['claim_recall']:.2f} ({r['reason']})"
            )
    print()
    print("NOTE: escalation gate fires only when both gain >= threshold and")
    print("full-text cost <= threshold. A '0 escalate' result means the existing")
    print("two-round retrieval already achieves the coverage floor, so full-text")
    print("reading is not worth its cost on this golden set.")


if __name__ == "__main__":
    main()
