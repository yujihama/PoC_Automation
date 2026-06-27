"""Create the v3 fixture with readable text procedures and visual evidence files."""

from __future__ import annotations

import json
import shutil
import struct
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "examples" / "v3_multimodal_human_ref"


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "procedures").mkdir(parents=True)
    (OUT_DIR / "evidence").mkdir(parents=True)

    cases = [_case_identity(), _case_invoice(), _case_employment(), _case_contract()]
    for case in cases:
        procedure_path = OUT_DIR / str(case["procedure_csv_path"])
        procedure_path.parent.mkdir(parents=True, exist_ok=True)
        procedure_path.write_text(str(case.pop("_procedure_text")), encoding="utf-8")

        evidence_dir = OUT_DIR / str(case["evidence_bundle_path"])
        evidence_dir.mkdir(parents=True, exist_ok=True)
        for artifact in case.pop("_artifacts"):
            path = evidence_dir / str(artifact["name"])
            if artifact["type"] == "pdf":
                _write_pdf(
                    path,
                    title=str(artifact["title"]),
                    lines=list(artifact["visual_lines"]),
                    page_count=int(artifact.get("pages", 1)),
                    content_page=int(artifact.get("content_page", 1)),
                )
            elif artifact["type"] == "bmp":
                _write_bmp(
                    path,
                    title=str(artifact["title"]),
                    lines=list(artifact["visual_lines"]),
                    palette=str(artifact["palette"]),
                )
            else:
                raise ValueError(f"unknown artifact type: {artifact['type']}")
            (evidence_dir / f"{artifact['name']}.txt").write_text(
                textwrap.dedent(str(artifact["sidecar"])).strip() + "\n",
                encoding="utf-8",
            )

    manifest = {
        "dataset_id": "v3_multimodal_human_reference_dataset",
        "metadata": {
            "description": (
                "v3検証用。自然言語手続、PDF/画像証跡、非構造の人間実施結果から、"
                "ケース横断で追加指示を探索する。"
            ),
            "source": "scripts/create_v3_multimodal_fixture.py",
        },
        "cases": cases,
    }
    (OUT_DIR / "dataset.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"created {OUT_DIR / 'dataset.json'}")


def _case_identity() -> dict[str, object]:
    human_text = """\
    このケースは適合と判断しました。まず identity_summary.pdf の1ページ目にある「本人確認記録」欄で、氏名が佐藤理央、住所が東京都品川区東品川2-2-24、確認日が2026年5月31日と読めます。次に application_capture.bmp のOCR欄では、申請者名が同じ佐藤理央で、連絡先住所も東京都品川区東品川2-2-24になっていました。手続は申請内容と証跡の突合、必須項目の不足確認、判断根拠の記録を求めているため、氏名と住所の一致が主要な確認点です。PDF下部の備考には「転送不可郵便で到達確認済」とあり、住所が単なる自己申告だけではない点も補強材料になります。画像側にある受付メモではマンション名が省略されていますが、番地まで一致しており、PDF側にも部屋番号は任意項目と記載されています。確認日は古すぎず、手続上の有効期間内に収まるため、日付を理由に保留する必要もありません。また、別名義や旧住所を示す反対証跡は添付ファイル内に見当たりません。このため住所不一致とは扱わず、確認済みの証跡に基づいて適合とします。根拠はPDF上段の本人確認記録と、画像中央の申請者情報欄です。"""
    _assert_human_text_length("case_mm_001", human_text)
    return {
        "case_id": "case_mm_001",
        "split": "train",
        "procedure_csv_path": "procedures/case_mm_001_procedure.txt",
        "evidence_bundle_path": "evidence/case_mm_001",
        "_procedure_text": (
            "1. 申請内容と証跡を照合する。2. 氏名・住所・日付の不足や不一致を確認する。"
            "3. 判断と根拠箇所を短く記録する。"
        ),
        "_artifacts": [
            {
                "type": "pdf",
                "name": "identity_summary.pdf",
                "title": "IDENTITY SUMMARY",
                "visual_lines": [
                    "EVIDENCE ID: mm_identity_summary",
                    "SECTION: Identity Verification Record",
                    "NAME: SATO RIO",
                    "ADDRESS: Tokyo Shinagawa Higashi-Shinagawa 2-2-24",
                    "CHECK DATE: 2026-05-31",
                    "NOTE: Non-forwarding mail delivery confirmed.",
                    "ROOM NUMBER: Optional field.",
                ],
                "sidecar": """
                evidence_id: mm_identity_summary
                page: 1
                section: 本人確認記録
                氏名: 佐藤理央
                住所: 東京都品川区東品川2-2-24
                確認日: 2026-05-31
                備考: 転送不可郵便で到達確認済。部屋番号は任意項目。
                """,
            },
            {
                "type": "bmp",
                "name": "application_capture.bmp",
                "title": "APPLICATION CAPTURE",
                "palette": "green",
                "visual_lines": [
                    "APPLICANT: SATO RIO",
                    "CONTACT ADDRESS: Tokyo Shinagawa Higashi-Shinagawa 2-2-24",
                    "INTAKE MEMO: Building name omitted.",
                    "MATCH: Same name and street number as identity summary.",
                ],
                "sidecar": """
                evidence_id: mm_identity_application
                image region: 中央の申請者情報欄
                申請者名: 佐藤理央
                連絡先住所: 東京都品川区東品川2-2-24
                受付メモ: マンション名は未入力。本人確認票と番地まで一致。
                """,
            },
        ],
        "human_result_text": textwrap.dedent(human_text).strip(),
        "expected_output": {
            "judgement": "適合",
            "required_claim_keywords": ["氏名", "住所", "一致"],
            "citations": [{"evidence_id": "mm_identity_summary", "page": 1, "claim": "本人確認記録の氏名住所"}],
        },
        "metadata": {
            "domain": "本人確認",
            "procedure_family": "本人属性照合",
            "difficulty": "medium",
            "required_capability": "evidence_grounding",
        },
    }


def _case_invoice() -> dict[str, object]:
    human_text = """\
    このケースは不適合としました。billing_invoice.pdf の1ページ目、請求明細の合計欄には税抜120,000円、消費税12,000円、税込132,000円とあり、支払依頼額は132,000円です。一方で order_screen.bmp の右上の発注情報では、承認済み発注額が123,000円、許容差額は1,000円までと読めます。手続は金額、日付、承認状態を照合して差異を記録する内容なので、請求書が発注額を9,000円超過している点は単なる丸め誤差ではありません。PDFの備考に「追加作業を含む」とありますが、画像側の発注画面には追加作業の変更承認番号が空欄で、承認済み金額を増やす根拠にはできません。請求日と発注日は同じ月内で、承認状態も発注側は承認済みですが、最終的な支払可否では金額不一致が重大です。追加作業を認めるには変更承認または別発注が必要ですが、今回の証跡にはその箇所が見当たりません。税抜金額だけを見ると発注額に近く見えるものの、手続では支払依頼額を比較するため税込合計を採用します。これは重要です。根拠はPDF下部の合計欄と、画像右上の発注額・許容差額欄を見比べた結果です。"""
    _assert_human_text_length("case_mm_002", human_text)
    return {
        "case_id": "case_mm_002",
        "split": "train",
        "procedure_csv_path": "procedures/case_mm_002_procedure.txt",
        "evidence_bundle_path": "evidence/case_mm_002",
        "_procedure_text": (
            "1. 請求書と発注情報を照合する。2. 金額・日付・承認状態の差異を確認する。"
            "3. 許容差を超える場合は根拠付きで不適合にする。"
        ),
        "_artifacts": [
            {
                "type": "pdf",
                "name": "billing_invoice.pdf",
                "title": "BILLING INVOICE",
                "visual_lines": [
                    "EVIDENCE ID: mm_billing_invoice",
                    "SECTION: Invoice Detail",
                    "NET AMOUNT: 120000",
                    "TAX: 12000",
                    "TOTAL REQUEST: 132000",
                    "NOTE: Includes additional work.",
                ],
                "sidecar": """
                evidence_id: mm_billing_invoice
                page: 1
                section: 請求明細
                税抜金額: 120000
                消費税: 12000
                税込合計: 132000
                備考: 追加作業を含む。
                """,
            },
            {
                "type": "bmp",
                "name": "order_screen.bmp",
                "title": "ORDER SCREEN",
                "palette": "red",
                "visual_lines": [
                    "APPROVED ORDER AMOUNT: 123000",
                    "ALLOWED DIFFERENCE: 1000",
                    "APPROVAL STATUS: Approved",
                    "CHANGE APPROVAL NO: Blank",
                ],
                "sidecar": """
                evidence_id: mm_order_screen
                image region: 右上の発注情報
                承認済み発注額: 123000
                許容差額: 1000
                承認状態: 承認済み
                変更承認番号: 空欄
                """,
            },
        ],
        "human_result_text": textwrap.dedent(human_text).strip(),
        "expected_output": {
            "judgement": "不適合",
            "required_claim_keywords": ["金額", "不一致", "132000"],
            "citations": [{"evidence_id": "mm_billing_invoice", "page": 1, "claim": "請求額132000円"}],
        },
        "metadata": {
            "domain": "請求審査",
            "procedure_family": "金額照合",
            "difficulty": "hard",
            "required_capability": "citation_precision",
        },
    }


def _case_employment() -> dict[str, object]:
    human_text = """\
    このケースは判断不能です。employment_certificate.pdf の1ページ目には、勤務先が北浜製作所、雇用区分が契約社員、在籍状態が在籍中とあります。ただし、手続で確認対象になっている発行日がPDFの右上欄では空欄です。applicant_note.bmp の左下にある申告メモでは「2026年4月以降も勤務継続」と書かれていますが、これは本人申告であり、証明書の発行日を補う証跡ではありません。手続は雇用区分と在籍状態に加え、証跡の有効期間を確認するとしているため、発行日がないまま現在性を判断するのは危険です。PDF本文に押印らしき表示はありますが、OCR欄にも発行日がなく、画像側にも会社発行の別資料はありません。契約社員であること自体は不適合理由ではないものの、証跡の発行時点が特定できないため、適合にも不適合にも寄せず、追加証跡が必要な判断不能とします。仮に在籍中という文言だけを見ると適合に寄りやすいですが、今回は有効期間確認が手続の独立した条件です。古い証明書の再利用可能性を排除できない点も重視しました。根拠はPDF右上の発行日空欄と、画像左下の本人申告メモの性質です。"""
    _assert_human_text_length("case_mm_003", human_text)
    return {
        "case_id": "case_mm_003",
        "split": "validation",
        "procedure_csv_path": "procedures/case_mm_003_procedure.txt",
        "evidence_bundle_path": "evidence/case_mm_003",
        "_procedure_text": (
            "1. 雇用証明の勤務先・雇用区分・在籍状態を確認する。2. 発行日と有効期間を見る。"
            "3. 現在性が確認できなければ判断不能にする。"
        ),
        "_artifacts": [
            {
                "type": "pdf",
                "name": "employment_certificate.pdf",
                "title": "EMPLOYMENT CERTIFICATE",
                "visual_lines": [
                    "EVIDENCE ID: mm_employment_certificate",
                    "SECTION: Employment Certificate",
                    "EMPLOYER: Kitahama Manufacturing",
                    "EMPLOYMENT TYPE: Contract employee",
                    "STATUS: Active",
                    "ISSUE DATE: Blank",
                ],
                "sidecar": """
                evidence_id: mm_employment_certificate
                page: 1
                section: 在籍証明書
                勤務先: 北浜製作所
                雇用区分: 契約社員
                在籍状態: 在籍中
                発行日: 空欄
                """,
            },
            {
                "type": "bmp",
                "name": "applicant_note.bmp",
                "title": "APPLICANT NOTE",
                "palette": "blue",
                "visual_lines": [
                    "SELF-REPORTED NOTE:",
                    "Continued working after 2026-04.",
                    "ATTACHED COMPANY SUPPLEMENT: None",
                    "SOURCE TYPE: Applicant memo, not employer proof.",
                ],
                "sidecar": """
                evidence_id: mm_applicant_note
                image region: 左下の申告メモ
                本人申告: 2026年4月以降も勤務継続
                添付資料: 会社発行の補足資料なし
                """,
            },
        ],
        "human_result_text": textwrap.dedent(human_text).strip(),
        "expected_output": {
            "judgement": "判断不能",
            "required_claim_keywords": ["発行日", "空欄", "判断不能"],
            "citations": [{"evidence_id": "mm_employment_certificate", "page": 1, "claim": "発行日空欄"}],
        },
        "metadata": {
            "domain": "雇用確認",
            "procedure_family": "証跡不足判定",
            "difficulty": "hard",
            "required_capability": "abstention",
        },
    }


def _case_contract() -> dict[str, object]:
    human_text = """\
    このケースは適合と判断しました。contract_terms.pdf の2ページ目の例外条件欄には「月次売上が300万円以上、かつ解約通知が30日前までに提出されている場合、通常の最低利用期間を適用しない」とあります。sales_dashboard.bmp の中央の月次集計では対象月の売上が3,180,000円で、notice_scan.pdf の1ページ目には解約通知の受領日が2026年5月1日、終了希望日が2026年6月15日と読めます。30日前条件は45日前に相当するので満たしています。手続は原則条件だけでなく例外条件の成立有無を確認することを求めており、最低利用期間だけを見ると不適合に見えますが、例外条件まで読むと適合です。画像の売上は速報値ではなく「確定」と表示され、PDFの通知書にも受付印があります。契約条項、売上画面、通知書の三つがそろって初めて例外成立といえるため、一つだけを根拠にした判断では不十分です。売上額は300万円を18万円上回っており、境界値の読み違いでもありません。したがって、根拠は契約条項2ページの例外条件、画像中央の確定売上、通知書PDFの受領日と終了希望日の組み合わせです。"""
    _assert_human_text_length("case_mm_004", human_text)
    return {
        "case_id": "case_mm_004",
        "split": "holdout",
        "procedure_csv_path": "procedures/case_mm_004_procedure.txt",
        "evidence_bundle_path": "evidence/case_mm_004",
        "_procedure_text": (
            "1. 契約の原則条件と例外条件を読む。2. 売上・通知日など成立要件を証跡で確認する。"
            "3. 例外が成立する場合は根拠を明記する。"
        ),
        "_artifacts": [
            {
                "type": "pdf",
                "name": "contract_terms.pdf",
                "title": "CONTRACT TERMS",
                "pages": 2,
                "content_page": 2,
                "visual_lines": [
                    "EVIDENCE ID: mm_contract_terms",
                    "PAGE: 2",
                    "SECTION: Exception Conditions",
                    "Monthly sales >= 3,000,000 JPY",
                    "AND cancellation notice submitted at least 30 days before end date.",
                    "If both are met, minimum usage period does not apply.",
                ],
                "sidecar": """
                evidence_id: mm_contract_terms
                page: 2
                section: 例外条件
                条項: 月次売上が300万円以上、かつ解約通知が30日前までに提出されている場合、通常の最低利用期間を適用しない。
                """,
            },
            {
                "type": "bmp",
                "name": "sales_dashboard.bmp",
                "title": "SALES DASHBOARD",
                "palette": "orange",
                "visual_lines": [
                    "MONTHLY SALES: 3180000",
                    "THRESHOLD: 3000000",
                    "AGGREGATION STATUS: Final",
                    "RESULT: Above threshold by 180000",
                ],
                "sidecar": """
                evidence_id: mm_sales_dashboard
                image region: 中央の月次集計
                対象月売上: 3180000
                集計状態: 確定
                """,
            },
            {
                "type": "pdf",
                "name": "notice_scan.pdf",
                "title": "NOTICE SCAN",
                "visual_lines": [
                    "EVIDENCE ID: mm_notice_scan",
                    "SECTION: Cancellation Notice",
                    "RECEIVED DATE: 2026-05-01",
                    "REQUESTED END DATE: 2026-06-15",
                    "DAYS BEFORE END DATE: 45",
                    "RECEPTION STAMP: Present",
                ],
                "sidecar": """
                evidence_id: mm_notice_scan
                page: 1
                section: 解約通知書
                受領日: 2026-05-01
                終了希望日: 2026-06-15
                受付印: あり
                """,
            },
        ],
        "human_result_text": textwrap.dedent(human_text).strip(),
        "expected_output": {
            "judgement": "適合",
            "required_claim_keywords": ["例外条件", "売上", "通知"],
            "citations": [{"evidence_id": "mm_contract_terms", "page": 2, "claim": "例外条件"}],
        },
        "metadata": {
            "domain": "契約審査",
            "procedure_family": "条件分岐判定",
            "difficulty": "hard",
            "required_capability": "condition_check",
        },
    }


def _write_pdf(path: Path, *, title: str, lines: list[str], page_count: int = 1, content_page: int = 1) -> None:
    page_count = max(1, page_count)
    content_page = min(max(1, content_page), page_count)

    objects: list[str] = [
        "<< /Type /Catalog /Pages 2 0 R >>\n",
        "",  # pages object is filled after page ids are known
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n",
    ]
    page_ids: list[int] = []
    for page_no in range(1, page_count + 1):
        page_id = len(objects) + 1
        content_id = page_id + 1
        page_ids.append(page_id)
        content = _pdf_page_content(
            title=title,
            lines=lines if page_no == content_page else [f"See page {content_page} for evidence details."],
            page_no=page_no,
            page_count=page_count,
        )
        stream = f"<< /Length {len(content.encode('ascii'))} >>\nstream\n{content}endstream\n"
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>\n"
        )
        objects.append(stream)
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>\n"
    _write_pdf_objects(path, objects)


def _pdf_page_content(*, title: str, lines: list[str], page_no: int, page_count: int) -> str:
    commands = [
        "q 0.93 0.96 1.00 rg 48 708 516 44 re f Q\n",
        "q 0.78 0.86 0.96 RG 48 708 516 44 re S Q\n",
        _pdf_text(title, 64, 724, 18),
        _pdf_text(f"Page {page_no} / {page_count}", 480, 724, 10),
        "q 0.15 0.15 0.15 RG 48 684 516 0.8 re f Q\n",
    ]
    y = 650
    for idx, line in enumerate(lines):
        if idx % 2 == 0:
            commands.append("q 0.97 0.97 0.97 rg 58 " + str(y - 7) + " 496 24 re f Q\n")
        commands.append(_pdf_text(line[:92], 68, y, 11))
        y -= 30
    return "".join(commands)


def _pdf_text(text: str, x: int, y: int, size: int) -> str:
    safe = _pdf_escape(_ascii(text))
    return f"BT /F1 {size} Tf {x} {y} Td ({safe}) Tj ET\n"


def _write_pdf_objects(path: Path, objects: list[str]) -> None:
    payload = b"%PDF-1.4\n"
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload += f"{idx} 0 obj\n".encode("ascii") + obj.encode("ascii") + b"endobj\n"
    xref_offset = len(payload)
    xref = f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    xref += "".join(f"{offset:010d} 00000 n \n" for offset in offsets[1:])
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    path.write_bytes(payload + xref.encode("ascii") + trailer.encode("ascii"))


def _write_bmp(path: Path, *, title: str, lines: list[str], palette: str) -> None:
    width = 900
    height = 520
    palette_map = {
        "green": ((236, 249, 242), (45, 129, 92), (22, 80, 62), (255, 255, 255)),
        "red": ((255, 239, 236), (185, 67, 54), (112, 47, 42), (255, 255, 255)),
        "blue": ((236, 242, 255), (72, 101, 181), (35, 54, 118), (255, 255, 255)),
        "orange": ((255, 244, 226), (205, 128, 44), (123, 75, 32), (255, 255, 255)),
    }
    bg, accent, dark, white = palette_map[palette]
    image = [[bg for _ in range(width)] for _ in range(height)]
    _rect(image, 36, 32, width - 72, 76, accent)
    _text(image, _ascii(title), 62, 58, white, scale=4)
    _rect(image, 54, 134, width - 108, 300, white)
    _rect_outline(image, 54, 134, width - 108, 300, accent)
    y = 165
    for line in lines:
        _text(image, _ascii(line[:70]), 86, y, dark, scale=3)
        y += 56
    _rect(image, 54, 462, width - 108, 24, accent)
    _text(image, "OCR SIDECAR AVAILABLE: SEE MATCHING .TXT FILE", 86, 466, white, scale=2)
    _write_bmp_pixels(path, image)


def _write_bmp_pixels(path: Path, image: list[list[tuple[int, int, int]]]) -> None:
    height = len(image)
    width = len(image[0])
    row_size = (width * 3 + 3) & ~3
    pixels = bytearray()
    for row in reversed(image):
        data = bytearray()
        for r, g, b in row:
            data.extend((b, g, r))
        data.extend(b"\x00" * (row_size - width * 3))
        pixels.extend(data)
    file_size = 14 + 40 + len(pixels)
    header = b"BM" + struct.pack("<IHHI", file_size, 0, 0, 54)
    dib = struct.pack("<IIIHHIIIIII", 40, width, height, 1, 24, 0, len(pixels), 2835, 2835, 0, 0)
    path.write_bytes(header + dib + pixels)


def _rect(image: list[list[tuple[int, int, int]]], x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
    for yy in range(max(0, y), min(len(image), y + h)):
        row = image[yy]
        for xx in range(max(0, x), min(len(row), x + w)):
            row[xx] = color


def _rect_outline(
    image: list[list[tuple[int, int, int]]],
    x: int,
    y: int,
    w: int,
    h: int,
    color: tuple[int, int, int],
) -> None:
    _rect(image, x, y, w, 3, color)
    _rect(image, x, y + h - 3, w, 3, color)
    _rect(image, x, y, 3, h, color)
    _rect(image, x + w - 3, y, 3, h, color)


def _text(
    image: list[list[tuple[int, int, int]]],
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    *,
    scale: int,
) -> None:
    cursor = x
    for char in text.upper():
        glyph = FONT.get(char, FONT[" "])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    _rect(image, cursor + gx * scale, y + gy * scale, scale, scale, color)
        cursor += 6 * scale


def _ascii(text: str) -> str:
    return text.encode("ascii", errors="ignore").decode("ascii")


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _assert_human_text_length(case_id: str, text: str) -> None:
    length = len(textwrap.dedent(text).strip())
    if not 500 <= length <= 1000:
        raise ValueError(f"{case_id} human_result_text must be 500-1000 chars, got {length}")


FONT = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ",": ["00000", "00000", "00000", "00000", "00000", "01100", "01000"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "/": ["00001", "00010", "00100", "01000", "10000", "00000", "00000"],
    ">": ["10000", "01000", "00100", "00010", "00100", "01000", "10000"],
    "=": ["00000", "11111", "00000", "11111", "00000", "00000", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


if __name__ == "__main__":
    main()
