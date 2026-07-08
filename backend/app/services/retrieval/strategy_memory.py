"""Experience-RAG strategy memory with PII-safe JSONL persistence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from app.services.utils import pii_scrubber


def fingerprint_query(query: str) -> str:
    scrubbed = pii_scrubber((query or "").strip()).lower()
    return hashlib.sha256(scrubbed.encode("utf-8")).hexdigest()[:24]


class StrategyExperience(BaseModel):
    query_fingerprint: str
    query_traits: list[str]
    strategy_id: str
    retrieval_mode: str
    bm25_weight: float
    dense_weight: float
    top_k: int
    evidence_count: int
    citation_supported_rate: float | None = None
    unsupported_claim_rate: float | None = None
    source_precision: float | None = None
    human_review_outcome: str | None = None
    created_at: str

    @property
    def quality_score(self) -> float:
        supported = self.citation_supported_rate if self.citation_supported_rate is not None else 0.5
        precision = self.source_precision if self.source_precision is not None else 0.5
        unsupported = self.unsupported_claim_rate if self.unsupported_claim_rate is not None else 0.5
        return round((supported * 0.45) + (precision * 0.45) + ((1.0 - unsupported) * 0.10), 4)


class StrategyExperienceStore:
    def __init__(self, path: str | Path, max_records: int = 1000):
        self.path = Path(path)
        self.max_records = max(1, int(max_records))

    def _read_all(self) -> list[StrategyExperience]:
        if not self.path.exists():
            return []
        records: list[StrategyExperience] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(StrategyExperience(**json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
        return records[-self.max_records :]

    def record(
        self,
        *,
        query: str,
        query_traits: list[str],
        strategy_id: str,
        retrieval_mode: str,
        bm25_weight: float,
        dense_weight: float,
        top_k: int,
        evidence_count: int,
        citation_supported_rate: float | None = None,
        unsupported_claim_rate: float | None = None,
        source_precision: float | None = None,
        human_review_outcome: str | None = None,
    ) -> StrategyExperience:
        scrubbed_query = pii_scrubber(query)
        record = StrategyExperience(
            query_fingerprint=fingerprint_query(scrubbed_query),
            query_traits=sorted({trait.lower() for trait in query_traits if trait}),
            strategy_id=strategy_id,
            retrieval_mode=retrieval_mode,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
            top_k=top_k,
            evidence_count=evidence_count,
            citation_supported_rate=citation_supported_rate,
            unsupported_claim_rate=unsupported_claim_rate,
            source_precision=source_precision,
            human_review_outcome=human_review_outcome,
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.model_dump(), ensure_ascii=False, sort_keys=True) + "\n")
        return record

    def find_similar(self, query_traits: list[str], limit: int = 5) -> list[StrategyExperience]:
        wanted = {trait.lower() for trait in query_traits if trait}
        scored: list[tuple[int, float, StrategyExperience]] = []
        for record in self._read_all():
            overlap = len(wanted.intersection({trait.lower() for trait in record.query_traits}))
            if overlap:
                scored.append((overlap, record.quality_score, record))
        scored.sort(key=lambda item: (item[0], item[1], item[2].created_at), reverse=True)
        return [record for _, _, record in scored[:limit]]
