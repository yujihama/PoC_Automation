from pathlib import Path

from poc_automation.evidence import read_text_artifact


def test_read_text_artifact_uses_pdf_and_image_sidecars(tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "document.pdf").write_bytes(b"%PDF-1.4\n")
    (evidence_dir / "document.pdf.txt").write_text("evidence_id: pdf_doc\namount: 132000\n", encoding="utf-8")
    (evidence_dir / "screen.bmp").write_bytes(b"BMfake")
    (evidence_dir / "screen.bmp.txt").write_text("evidence_id: image_screen\namount: 123000\n", encoding="utf-8")

    text = read_text_artifact(evidence_dir, max_chars=5000)

    assert "## document.pdf" in text
    assert "evidence_id: pdf_doc" in text
    assert "## screen.bmp" in text
    assert "evidence_id: image_screen" in text
    assert "## document.pdf.txt" not in text
    assert "## screen.bmp.txt" not in text


def test_read_text_artifact_truncates_directory_output(tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "long.txt").write_text("a" * 200, encoding="utf-8")

    text = read_text_artifact(evidence_dir, max_chars=50)

    assert len(text) > 50
    assert text.endswith("...[truncated]")


def test_v3_fixture_visual_artifacts_are_not_blank():
    pdf_path = Path("examples/v3_multimodal_human_ref/evidence/case_mm_002/billing_invoice.pdf")
    image_path = Path("examples/v3_multimodal_human_ref/evidence/case_mm_002/order_screen.bmp")

    pdf_text = pdf_path.read_bytes().decode("latin-1")
    assert "TOTAL REQUEST" in pdf_text
    assert "132000" in pdf_text

    image_bytes = image_path.read_bytes()
    assert len(image_bytes) > 1_000_000
    pixel_payload = image_bytes[54:]
    assert len(set(pixel_payload[::101])) > 3
