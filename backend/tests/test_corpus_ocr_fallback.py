from pathlib import Path

from langchain_core.documents import Document

from app.schemas.corpus import SourceDocument
from app.services.corpus import corpus_ingestor


def _source(path: Path) -> SourceDocument:
    source = SourceDocument(
        doc_id="scanned_required",
        title="Scanned required source",
        regulator="PCPD",
        doc_type="Checklist",
        issue_date="2025-03",
        sector=["Privacy"],
        topics=["GenAI"],
        jurisdiction="Hong Kong",
        applicability=["demo"],
        file_path=path.name,
        priority="P1",
        language="en",
        source_url="https://www.pcpd.org.hk/example.pdf",
        status="active",
        effective_date="2025-03-01",
        required_for_demo=True,
    )
    source.resolved_path = path
    return source


class _ImageOnlyLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        return [Document(page_content="  ", metadata={"page": 0})]


class _TextLoader(_ImageOnlyLoader):
    def load(self):
        return [Document(page_content="Native extracted text", metadata={"page": 0})]


def test_image_only_pdf_falls_back_to_ocr(tmp_path):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-test")
    calls = []

    def ocr_loader(path):
        calls.append(path)
        return [Document(page_content="OCR extracted controls", metadata={"page": 0, "parser": "ocr"})]

    pages = corpus_ingestor._load_source_pages(
        _source(pdf),
        pdf_loader_cls=_ImageOnlyLoader,
        ocr_loader=ocr_loader,
    )

    assert calls == [pdf]
    assert pages[0].page_content == "OCR extracted controls"
    assert pages[0].metadata["parser"] == "ocr"


def test_text_pdf_does_not_invoke_ocr(tmp_path):
    pdf = tmp_path / "text.pdf"
    pdf.write_bytes(b"%PDF-test")

    pages = corpus_ingestor._load_source_pages(
        _source(pdf),
        pdf_loader_cls=_TextLoader,
        ocr_loader=lambda path: (_ for _ in ()).throw(AssertionError("OCR should not run")),
    )

    assert pages[0].page_content == "Native extracted text"


def test_ocr_pdf_pages_preserves_page_numbers(monkeypatch, tmp_path):
    class FakeBitmap:
        def to_numpy(self):
            return object()

    class FakePage:
        def __init__(self, index):
            self.index = index

        def render(self, scale):
            assert scale == 2.0
            return FakeBitmap()

    class FakePdf:
        def __init__(self, path):
            self.pages = [FakePage(0), FakePage(1)]

        def __len__(self):
            return len(self.pages)

        def __getitem__(self, index):
            return self.pages[index]

        def close(self):
            pass

    class FakeResult:
        def __init__(self):
            self.txts = ("first line", "second line")

    monkeypatch.setattr(corpus_ingestor, "_build_pdf_document", lambda path: FakePdf(path))
    monkeypatch.setattr(corpus_ingestor, "_build_ocr_engine", lambda: lambda image: FakeResult())

    pages = corpus_ingestor._ocr_pdf_pages(tmp_path / "scan.pdf")

    assert [page.metadata["page"] for page in pages] == [0, 1]
    assert all(page.metadata["parser"] == "rapidocr" for page in pages)
    assert pages[0].page_content == "first line\nsecond line"
