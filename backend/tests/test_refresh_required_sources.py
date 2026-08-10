from io import BytesIO

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.schemas.corpus import SourceDocument
from scripts.refresh_required_sources import refresh_source


def _source(file_path: str = "source.pdf", source_url: str = "https://www.hkma.gov.hk/source.pdf"):
    return SourceDocument(
        doc_id="required_source",
        title="Required regulatory source",
        regulator="HKMA",
        doc_type="Guideline",
        file_path=file_path,
        source_url=source_url,
        status="active",
        metadata_note="Effective date is not separately stated.",
        required_for_demo=True,
    )


def _valid_pdf_bytes() -> bytes:
    output = BytesIO()
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )
    content = DecodedStreamObject()
    content.set_data(b"BT /F1 12 Tf 72 720 Td (Regulatory source text) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(content)
    writer.write(output)
    return output.getvalue()


def test_invalid_download_never_replaces_existing_file(tmp_path):
    destination = tmp_path / "source.pdf"
    destination.write_bytes(b"original")

    with pytest.raises(ValueError, match="valid PDF"):
        refresh_source(_source(), tmp_path, opener=lambda _request: BytesIO(b"<html>not a pdf</html>"))

    assert destination.read_bytes() == b"original"


def test_valid_download_is_atomically_installed(tmp_path):
    destination = refresh_source(
        _source(),
        tmp_path,
        opener=lambda _request: BytesIO(_valid_pdf_bytes()),
    )

    reader = PdfReader(str(destination))
    assert len(reader.pages) == 1
    assert "Regulatory source text" in reader.pages[0].extract_text()


def test_non_official_url_is_rejected_before_network_access(tmp_path):
    def unexpected_network(_request):
        raise AssertionError("invalid source metadata must fail before download")

    with pytest.raises(ValueError, match="official regulator domain"):
        refresh_source(
            _source(source_url="https://example.com/source.pdf"),
            tmp_path,
            opener=unexpected_network,
        )
