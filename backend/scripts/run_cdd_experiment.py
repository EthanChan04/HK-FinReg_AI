"""T3-02: Run CDD conflict diagnosis on real corpus-derived cases."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.retrieval.cdd_diagnoser import diagnose_conflicts

# Case 1: 正确法规 vs 过时法规（SFC 适当性：FAQ 要求评估客户风险态度）
# Claims use the corpus's own wording (a faithful generator restates evidence).
CASE_CORRECT_CONTEXT = [
    "Licensed or registered persons should make an assessment of the client's attitude towards risk, his expectations and so on based on the information disclosed by a client.",
    "Each client's information should be properly documented and where appropriate, updated on a continuous basis.",
]
CASE_STALE_PRIOR = [
    "Licensed persons were permitted to rely solely on the client's stated investment objectives without an independent risk assessment.",
]
CLAIMS_1 = [
    "Licensed or registered persons should make an assessment of the client's attitude towards risk.",
    "Licensed persons were permitted to rely solely on the client's stated investment objectives.",
]

# Case 2: 正确法规 vs 误导性摘要（PCPD AI 框架：C-level 指定）
CASE_CORRECT_CONTEXT_2 = [
    "A C-level executive should be designated to steer the implementation of the AI strategy and oversee the procurement, implementation and use of AI systems.",
]
CASE_MISLEADING_PRIOR_2 = [
    "Line managers may self-certify AI governance without any senior management oversight.",
]
CLAIMS_2 = [
    "A C-level executive should be designated to steer the implementation of the AI strategy.",
    "Line managers may self-certify AI governance without any senior management oversight.",
]

# Case 3: 无冲突基线（正确法规 vs 一致先验）
CASE_CORRECT_CONTEXT_3 = [
    "Authorized institutions should provide an appropriate level of transparency to customers regarding their GenAI applications.",
]
CASE_CONSISTENT_PRIOR_3 = [
    "Authorized institutions should provide an appropriate level of transparency to customers.",
]
CLAIMS_3 = [
    "Authorized institutions should provide an appropriate level of transparency to customers regarding their GenAI applications.",
]

for name, ctx, prior, claims, ground_truth in [
    ("正确法规 vs 过时法规", CASE_CORRECT_CONTEXT, CASE_STALE_PRIOR, CLAIMS_1, [CLAIMS_1[1]]),
    ("正确法规 vs 误导性摘要", CASE_CORRECT_CONTEXT_2, CASE_MISLEADING_PRIOR_2, CLAIMS_2, [CLAIMS_2[1]]),
    ("无冲突基线", CASE_CORRECT_CONTEXT_3, CASE_CONSISTENT_PRIOR_3, CLAIMS_3, []),
]:
    report = diagnose_conflicts(
        claims,
        ctx,
        prior,
        conflicting_claims=ground_truth,
    )
    print(f"=== {name} ===")
    print(f"  检测率={report.conflict_detection_rate} 误报率={report.false_positive_rate} 总数={report.total_claims}")
    for d in report.diagnoses:
        print(f"  [{'冲突' if d.conflict else '一致'}] ctx={d.context_supported} prior={d.prior_supported} | {d.claim[:70]}")
    print()
