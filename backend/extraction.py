from pathlib import Path


def extract_text(path: str | Path) -> str:
    """Extract text from a PDF, DOCX, or plain text file."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(file_path)).pages)
    if suffix == ".docx":
        from docx import Document

        return "\n".join(paragraph.text for paragraph in Document(str(file_path)).paragraphs)
    raise ValueError("Supported document types are PDF, DOCX, and TXT.")


def extract_pages(path: str | Path) -> list[str] | None:
    """Return PDF page text when page boundaries are available."""
    file_path = Path(path)
    if file_path.suffix.lower() != ".pdf":
        return None
    from pypdf import PdfReader

    return [page.extract_text() or "" for page in PdfReader(str(file_path)).pages]