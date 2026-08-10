"""Safely refresh required regulatory PDFs from official source URLs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.corpus import SourceDocument
from app.services.corpus.manifest_loader import (
    validate_manifest_release_gate,
    validate_source_metadata,
)


MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024


def _download(source: SourceDocument, opener) -> bytes:
    request = Request(
        str(source.source_url),
        headers={"User-Agent": "HK-FinReg-AI-corpus-refresh/1.0"},
    )
    response = opener(request, timeout=60) if opener is urlopen else opener(request)
    with response:
        final_url = getattr(response, "geturl", lambda: source.source_url)()
        if final_url and final_url != source.source_url:
            validate_source_metadata(
                {**source.model_dump(), "source_url": final_url}
            )
        content = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(content) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"{source.doc_id} download exceeds 50 MiB limit")
    return content


def _validate_pdf(path: Path, source: SourceDocument) -> None:
    try:
        reader = PdfReader(str(path))
        if not reader.pages:
            raise ValueError("PDF has no pages")
        if not any((page.extract_text() or "").strip() for page in reader.pages):
            raise ValueError("PDF contains no readable text")
    except Exception as exc:
        raise ValueError(f"{source.doc_id} download is not a valid PDF: {exc}") from exc


def refresh_source(
    source: SourceDocument,
    destination_root: Path,
    opener=urlopen,
) -> Path:
    """Download, validate, and atomically install one required source."""

    validate_source_metadata(source.model_dump())
    if not source.required_for_demo:
        raise ValueError(f"{source.doc_id} is not marked required_for_demo")

    destination_root = Path(destination_root).resolve()
    destination = (destination_root / source.file_path).resolve()
    if destination_root not in destination.parents:
        raise ValueError(f"{source.doc_id} file_path escapes the destination root")

    downloaded = _download(source, opener)
    if not downloaded.startswith(b"%PDF-"):
        raise ValueError(f"{source.doc_id} download is not a valid PDF")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.stem}-",
            suffix=".pdf",
            delete=False,
        ) as handle:
            handle.write(downloaded)
            temporary = Path(handle.name)
        _validate_pdf(temporary, source)
        temporary.replace(destination)
        temporary = None
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", action="append", required=True, dest="doc_ids")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    backend_root = Path(__file__).resolve().parents[1]
    manifest_path = backend_root / "data" / "source_manifest.json"
    destination_root = backend_root / "data" / "regulations"
    sources = {source.doc_id: source for source in validate_manifest_release_gate(manifest_path)}

    unknown = [doc_id for doc_id in args.doc_ids if doc_id not in sources]
    if unknown:
        raise ValueError(f"Unknown source doc_id(s): {', '.join(unknown)}")

    for doc_id in args.doc_ids:
        destination = refresh_source(sources[doc_id], destination_root)
        print(f"Refreshed {doc_id}: {destination}")


if __name__ == "__main__":
    main()
