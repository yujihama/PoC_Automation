"""Evidence artifact text extraction helpers.

The prototype stores image/PDF evidence locally.  The target runner still needs
text to reason over, so image and PDF files can be paired with OCR/extraction
sidecars such as `document.pdf.txt` or `scan.ocr.txt`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def read_text_artifact(path: str | Path, *, max_chars: int) -> str:
    artifact_path = Path(path)
    if artifact_path.is_file():
        return truncate_text(read_one_artifact_file(artifact_path), max_chars)
    if not artifact_path.exists():
        return f"path does not exist: {path}"

    sections: list[str] = []
    for file_path in sorted(p for p in artifact_path.rglob("*") if p.is_file()):
        if file_path.name.startswith(".") or _is_sidecar(file_path):
            continue
        sections.append(f"## {file_path.relative_to(artifact_path)}\n{read_one_artifact_file(file_path)}")
    return truncate_text("\n\n".join(sections), max_chars)


def read_one_artifact_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        text = path.read_text(encoding="utf-8-sig")
        try:
            return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            return text
    if suffix in {".txt", ".md", ".csv"}:
        return path.read_text(encoding="utf-8-sig")
    if suffix == ".pdf":
        sidecar = _sidecar_text(path)
        if sidecar is not None:
            return f"[PDF extracted text from {sidecar.name}]\n{sidecar.read_text(encoding='utf-8-sig')}"
        return "[PDF text unavailable; add a .pdf.txt extraction sidecar]\n" + _best_effort_pdf_text(path)
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".ppm"}:
        sidecar = _sidecar_text(path)
        if sidecar is not None:
            return f"[Image OCR text from {sidecar.name}]\n{sidecar.read_text(encoding='utf-8-sig')}"
        return f"[Image evidence: {path.name}; OCR sidecar not found]"
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return f"{path.name}: binary evidence file is not readable without an extraction sidecar"


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _sidecar_text(path: Path) -> Path | None:
    candidates = [
        path.with_name(path.name + ".txt"),
        path.with_suffix(path.suffix + ".txt"),
        path.with_name(path.stem + ".ocr.txt"),
        path.with_name(path.stem + ".txt"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _is_sidecar(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".pdf.txt") or name.endswith(".png.txt") or name.endswith(".jpg.txt") or name.endswith(
        ".jpeg.txt"
    ) or name.endswith(".bmp.txt") or name.endswith(".ocr.txt")


def _best_effort_pdf_text(path: Path) -> str:
    # This is intentionally limited.  It can recover text from simple generated
    # PDFs, but production-quality extraction should use an explicit sidecar or
    # a real PDF/OCR pipeline.
    data = path.read_bytes().decode("latin-1", errors="ignore")
    chunks = re.findall(r"\(([^()]*)\)", data)
    if not chunks:
        return ""
    return "\n".join(chunk.replace(r"\(", "(").replace(r"\)", ")") for chunk in chunks)

