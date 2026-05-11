"""Load retrieval benchmark questions."""

from __future__ import annotations

import json
from pathlib import Path


def load_benchmark_questions(path: str | Path | None = None) -> list[dict]:
    """Load benchmark questions from backend/data/evaluation."""

    if path is None:
        backend_root = Path(__file__).resolve().parents[3]
        path = backend_root / "data" / "evaluation" / "benchmark_questions.json"
    return json.loads(Path(path).read_text(encoding="utf-8"))
