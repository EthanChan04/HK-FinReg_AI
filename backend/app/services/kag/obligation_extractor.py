"""Rule-based obligation/risk/control extraction from evidence text."""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.evidence import EvidenceChunk


OBLIGATION_PATTERNS: dict[str, list[str]] = {
    "CDD": ["customer due diligence", "cdd", "know your customer", "kyc"],
    "EDD": ["enhanced due diligence", "edd"],
    "STR": ["suspicious transaction report", "str"],
    "Record Keeping": ["record-keeping", "record keeping", "records should be kept"],
    "Human Oversight": ["human oversight", "human review", "manual review"],
    "AI Governance": ["ai governance", "governance strategy"],
    "Data Protection": ["personal data", "privacy", "pdpo"],
    "Ongoing Monitoring": ["ongoing monitoring", "transaction monitoring"],
    "Model Governance": ["model risk", "model validation", "model governance"],
    "Third-party Risk Control": ["third party", "outsourcing", "vendor risk"],
}

RISK_PATTERNS: dict[str, list[str]] = {
    "AML/CFT": ["money laundering", "terrorist financing", "aml", "cft", "mltf"],
    "Fraud": ["fraud", "anti-fraud", "suspicious"],
    "Data Privacy": ["personal data", "privacy", "pdpo"],
    "Model Risk": ["model risk", "ai model", "algorithmic", "genai", "ai"],
    "Cybersecurity": ["cybersecurity", "security", "cyber"],
}

CONTROL_HINTS: dict[str, str] = {
    "CDD": "identity verification",
    "EDD": "enhanced customer review",
    "STR": "suspicious transaction escalation",
    "Record Keeping": "audit trail retention",
    "Human Oversight": "manual review gate",
    "AI Governance": "ai governance policy",
    "Data Protection": "data minimisation and lawful basis controls",
    "Ongoing Monitoring": "risk-based transaction monitoring",
    "Model Governance": "model validation and drift monitoring",
    "Third-party Risk Control": "vendor due diligence and contract controls",
}


@dataclass
class ExtractedObligation:
    name: str
    risk_type: str
    control_hint: str
    source_chunk_id: str
    confidence: float


def _match_label(text: str, patterns: dict[str, list[str]], default: str = "Operational Risk") -> str:
    lowered = text.lower()
    for label, terms in patterns.items():
        if any(term in lowered for term in terms):
            return label
    return default


def extract_obligations_from_chunks(evidence_chunks: list[EvidenceChunk]) -> list[ExtractedObligation]:
    """Extract coarse obligation records from retrieved evidence chunks."""

    extracted: list[ExtractedObligation] = []
    seen: set[tuple[str, str]] = set()
    for chunk in evidence_chunks:
        text = (chunk.text or "").strip()
        if not text:
            continue
        obligation = _match_label(text, OBLIGATION_PATTERNS, default="")
        if not obligation:
            continue
        risk = _match_label(text, RISK_PATTERNS, default="Operational Risk")
        if risk == "Operational Risk" and obligation in {"CDD", "EDD", "STR", "Ongoing Monitoring"}:
            risk = "AML/CFT"
        key = (chunk.evidence_id, obligation)
        if key in seen:
            continue
        seen.add(key)
        extracted.append(
            ExtractedObligation(
                name=obligation,
                risk_type=risk,
                control_hint=CONTROL_HINTS.get(obligation, "manual compliance control"),
                source_chunk_id=chunk.evidence_id,
                confidence=0.85,
            )
        )
    return extracted
