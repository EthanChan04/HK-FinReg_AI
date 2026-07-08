"""Corpus term statistics for SIRA-style query planning."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from pydantic import BaseModel, Field


def _normalize_term(term: str) -> str:
    return re.sub(r"\s+", " ", (term or "").strip().lower())


class TermStatistics(BaseModel):
    """Document-frequency metadata used to reject noisy query expansions."""

    document_count: int = 0
    document_frequency: dict[str, int] = Field(default_factory=dict)

    def df_ratio(self, term: str) -> float:
        if self.document_count <= 0:
            return 0.0
        return self.document_frequency.get(_normalize_term(term), 0) / self.document_count

    def is_allowed(
        self,
        term: str,
        *,
        query: str,
        metadata_terms: Iterable[str] = (),
        protected: bool = False,
        min_ratio: float = 0.002,
        max_ratio: float = 0.35,
    ) -> bool:
        """Return whether an expansion term is discriminative enough to use."""

        normalized = _normalize_term(term)
        if not normalized:
            return False
        query_l = (query or "").lower()
        metadata_l = {str(item).lower() for item in metadata_terms}
        if protected or normalized in query_l or normalized in metadata_l:
            return True
        ratio = self.df_ratio(normalized)
        if ratio > max_ratio:
            return False
        if ratio < min_ratio:
            return False
        return True

    @classmethod
    def from_documents(cls, documents: Iterable[Document]) -> "TermStatistics":
        docs = list(documents)
        counter: Counter[str] = Counter()
        for doc in docs:
            text = f"{getattr(doc, 'page_content', '')} {' '.join(str(v) for v in (doc.metadata or {}).values())}"
            terms = {
                _normalize_term(token)
                for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_ -]{1,40}", text)
                if token.strip()
            }
            counter.update(terms)
        return cls(document_count=len(docs), document_frequency=dict(counter))


def load_term_statistics(path: str | Path) -> TermStatistics | None:
    stats_path = Path(path)
    if not stats_path.exists():
        return None
    with stats_path.open("r", encoding="utf-8") as fh:
        return TermStatistics(**json.load(fh))


def save_term_statistics(stats: TermStatistics, path: str | Path) -> None:
    stats_path = Path(path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats.model_dump(), fh, ensure_ascii=False, indent=2, sort_keys=True)
