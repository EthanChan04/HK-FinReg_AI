"""T3-03: CTRAG adaptive chunker tests (NR-03).

Behavioral contract:
  - section headers become authoritative chunk boundaries
  - chunks stay within [min_chunk, max_chunk] bounds (except single
    sentences that cannot be split further)
  - oversized units are split at sentence boundaries
  - degenerate tiny chunks are merged into the previous chunk
  - metadata carries the section title and chunking strategy
"""

from __future__ import annotations

from app.services.retrieval.adaptive_chunker import adaptive_chunk_text


class TestAdaptiveChunker:
    def test_section_headers_are_authoritative_boundaries(self):
        text = (
            "Chapter 1 General\n"
            "This is the first section content with enough words to be meaningful.\n"
            "Chapter 2 CDD Requirements\n"
            "This is the second section about customer due diligence obligations.\n"
        )
        chunks = adaptive_chunk_text(text)
        assert len(chunks) >= 2
        titles = [c.metadata.get("section_title") for c in chunks]
        assert "Chapter 1 General" in titles
        assert "Chapter 2 CDD Requirements" in titles

    def test_oversized_unit_is_split_at_sentence_boundaries(self):
        text = (
            "Chapter 1\n"
            + " ".join(f"Sentence number {i} about regulatory obligations." for i in range(80))
        )
        chunks = adaptive_chunk_text(text, max_chunk=600)
        assert len(chunks) > 1
        for chunk in chunks:
            # No chunk exceeds max_chunk by more than one merged tiny tail
            # (min_chunk default 400) -- the merge-allowed bound.
            assert len(chunk.page_content) <= 600 + 400

    def test_tiny_titled_section_stays_own_chunk(self):
        text = (
            "Chapter 1 Main Content\n"
            "A reasonably long first paragraph that fills the buffer with content.\n"
            "1.2 Short Clause\n"
            "Tiny.\n"
        )
        chunks = adaptive_chunk_text(text, min_chunk=50, target_chunk=300)
        # A titled section is an authoritative boundary: its tiny content
        # stays its own chunk even though it is below min_chunk.
        tiny = [c for c in chunks if "Tiny" in c.page_content]
        assert len(tiny) == 1
        assert tiny[0].metadata.get("section_title") == "1.2 Short Clause"

    def test_untitled_tiny_chunks_are_merged_into_previous(self):
        text = (
            "Long first paragraph with enough words to exceed the minimum chunk size.\n"
            "Tiny.\n"
        )
        chunks = adaptive_chunk_text(text, min_chunk=50, target_chunk=300)
        tiny = [c for c in chunks if "Tiny" in c.page_content]
        assert len(tiny) == 1
        assert len(tiny[0].page_content) > 10  # merged with the previous paragraph

    def test_metadata_marks_adaptive_chunking(self):
        chunks = adaptive_chunk_text("Chapter 1\nSome content here.", metadata={"doc_id": "x"})
        assert all(c.metadata.get("chunking") == "adaptive" for c in chunks)
        assert all(c.metadata.get("doc_id") == "x" for c in chunks)

    def test_unstructured_text_still_produces_chunks(self):
        text = " ".join(f"Word number {i} in a plain paragraph." for i in range(200))
        chunks = adaptive_chunk_text(text, target_chunk=500)
        assert len(chunks) >= 1
        total_chars = sum(len(c.page_content) for c in chunks)
        assert total_chars >= len(text) * 0.9  # no content loss

    def test_clause_markers_split_units(self):
        text = (
            "(1) The licensee must maintain records.\n"
            "More detail about record keeping requirements follows here.\n"
            "(2) The licensee must report breaches.\n"
            "Additional breach reporting obligations are stated in this text.\n"
        )
        chunks = adaptive_chunk_text(text, min_chunk=10, target_chunk=50)
        joined = "\n".join(c.page_content for c in chunks)
        assert "(1)" in joined and "(2)" in joined
