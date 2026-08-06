"""T3-03: CTRAG-style adaptive chunking (NR-03, arXiv 2608.02472).

Adaptive chunking: chunk boundaries follow structural signals (section
headers, clause markers, paragraph breaks) and chunk size adapts to
content density instead of a fixed character window. Research prototype
for A/B comparison against the existing fixed-size splitter -- the
production pipeline is NOT modified.

Note on maturity: the CTRAG paper reports a POC at one Big-4 firm, not a
production deployment; treat gains as reference signals only.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document

# Structural signals: section headers, clause markers, numbered paragraphs.
# Headers may carry trailing title text ("Chapter 1 General").
_SECTION_HEADER = re.compile(
    r"^\s*(?:#+\s*)?(?:\d+\.\d*(?:\s+[A-Za-z][A-Za-z\s\-()]*)?|"
    r"(?:Chapter|Part|Section|Schedule|Annex|Appendix)\s+[A-Z0-9][A-Za-z0-9\s\-()]*|"
    r"(?:第\s*[一二三四五六七八九十\d]+\s*(?:章|节|部分|條|条)))[^\n]{0,40}\s*[:：]?\s*$",
    re.IGNORECASE,
)
_CLAUSE_MARKER = re.compile(r"^\s*(?:\(\d+\)|\d+\.\d+|\d+\)|[A-Za-z]\.\d+)\s+\S", re.IGNORECASE)

# Adaptive size bounds (chars). Dense regulatory prose gets smaller chunks,
# narrative paragraphs get larger ones.
MIN_CHUNK = 400
MAX_CHUNK = 2400
TARGET_CHUNK = 1200


def adaptive_chunk_text(
    text: str,
    metadata: dict | None = None,
    *,
    min_chunk: int = MIN_CHUNK,
    max_chunk: int = MAX_CHUNK,
    target_chunk: int = TARGET_CHUNK,
) -> list[Document]:
    """Split regulatory text with structure-aware, size-adaptive chunks.

    Strategy:
      1. Split on structural signals (section headers / clause markers)
         first -- these boundaries are authoritative and never crossed.
      2. Within a structural unit, accumulate paragraphs up to ``target``
         chars; a single oversized paragraph is split at sentence
         boundaries into ``max_chunk``-bounded pieces.
      3. Very short structural units are merged with the following unit
         until at least ``min_chunk`` chars (avoids degenerate chunks).
    """
    metadata = dict(metadata or {})
    units = _structural_units(text)
    if len(units) < 2:
        # Not a structured document: fall back to sentence-aware splitting.
        units = [(None, text)]

    docs: list[Document] = []
    buffer_title: str | None = None
    buffer = ""

    def flush() -> None:
        nonlocal buffer, buffer_title
        if buffer.strip():
            doc_meta = {**metadata}
            if buffer_title:
                doc_meta["section_title"] = buffer_title
            doc_meta["chunking"] = "adaptive"
            docs.append(Document(page_content=buffer.strip(), metadata=doc_meta))
        buffer = ""
        buffer_title = None

    for title, content in units:
        if len(content) <= target_chunk and title is None:
            # Untitled fragments may join the running buffer.
            if buffer and len(buffer) + len(content) > target_chunk:
                flush()
            if not buffer:
                buffer_title = title
            buffer += ("\n" if buffer else "") + content
            continue
        # Titled units are authoritative boundaries: never merged across.
        flush()
        if len(content) <= target_chunk:
            doc_meta = {**metadata}
            doc_meta["section_title"] = title
            doc_meta["chunking"] = "adaptive"
            docs.append(Document(page_content=content.strip(), metadata=doc_meta))
            continue
        for piece in _split_large_unit(content, max_chunk=max_chunk):
            doc_meta = {**metadata}
            if title:
                doc_meta["section_title"] = title
            doc_meta["chunking"] = "adaptive"
            docs.append(Document(page_content=piece.strip(), metadata=doc_meta))
    flush()

    # Merge degenerate tiny chunks (below min_chunk) into the previous one,
    # but never across a titled section boundary: a section header is an
    # authoritative chunk boundary and its (short) content stays its own chunk.
    final: list[Document] = []
    for doc in docs:
        prev = final[-1] if final else None
        merges_without_crossing_boundary = (
            prev is not None
            and len(doc.page_content) < min_chunk
            and (prev.metadata.get("section_title") == doc.metadata.get("section_title"))
        )
        if merges_without_crossing_boundary:
            final[-1] = Document(
                page_content=final[-1].page_content + "\n" + doc.page_content,
                metadata=final[-1].metadata,
            )
        else:
            final.append(doc)
    return final


def _structural_units(text: str) -> list[tuple[str | None, str]]:
    """Return [(header_title | None, content)] split on structural signals.

    Section headers become the unit title (header line itself is not part
    of the content). Clause markers stay IN the content and start a new
    unit, so the clause text itself is never lost.
    """
    lines = text.split("\n")
    units: list[tuple[str | None, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    def flush_unit() -> None:
        nonlocal current_lines
        if current_lines:
            units.append((current_title, "\n".join(current_lines)))
            current_lines = []

    for line in lines:
        stripped = line.strip()
        if _SECTION_HEADER.match(stripped):
            flush_unit()
            current_title = stripped
        elif _CLAUSE_MARKER.match(stripped):
            flush_unit()
            current_lines.append(line)
        else:
            current_lines.append(line)
    flush_unit()
    return units


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")


def _split_large_unit(content: str, *, max_chunk: int) -> list[str]:
    """Split an oversized structural unit at sentence boundaries."""
    sentences = [s for s in _SENTENCE_SPLIT.split(content) if s.strip()]
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > max_chunk and current:
            pieces.append(current)
            current = sentence
        else:
            current += (" " if current else "") + sentence
        # A single sentence longer than max_chunk gets hard-cut.
        if len(current) >= max_chunk:
            pieces.append(current)
            current = ""
    if current:
        pieces.append(current)
    return pieces or [content]
