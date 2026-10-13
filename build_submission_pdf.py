from __future__ import annotations

import argparse
import re
import subprocess
from html import escape
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT_SOURCE = ROOT / "EE6222_Final_Report_Public.md"
DEFAULT_PDF_PATH = ROOT / "EE6222_Assignment1_Report.pdf"
DEFAULT_AUTHOR = "JI YANCHEN"
APPENDIX_FILES = ("EE6222_Assignment_Standalone.py",)
FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\consola.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
)
MONO_FONT = "EE6222Code"


def register_fonts() -> None:
    font_path = next((path for path in FONT_CANDIDATES if path.is_file()), None)
    if font_path is None:
        candidates = ", ".join(str(path) for path in FONT_CANDIDATES)
        raise FileNotFoundError(f"Required code font not found. Checked: {candidates}")
    if MONO_FONT not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(MONO_FONT, str(font_path)))


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    black = colors.black
    return {
        "title": ParagraphStyle(
            "EE6222Title", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=15, leading=18, alignment=TA_CENTER, spaceAfter=5,
            textColor=black,
        ),
        "author": ParagraphStyle(
            "EE6222Author", parent=base["Normal"], fontName="Helvetica",
            fontSize=10.5, leading=12.5, alignment=TA_CENTER, spaceAfter=2,
            textColor=black,
        ),
        "section": ParagraphStyle(
            "EE6222Section", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=12, leading=14.5, spaceBefore=5, spaceAfter=4,
            textColor=black, keepWithNext=True,
        ),
        "subsection": ParagraphStyle(
            "EE6222Subsection", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=10.5, leading=12.7, spaceBefore=4, spaceAfter=2.5,
            textColor=black, keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "EE6222Body", parent=base["BodyText"], fontName="Times-Roman",
            fontSize=10.5, leading=13.1, alignment=TA_JUSTIFY, spaceAfter=6,
            textColor=black,
        ),
        "caption": ParagraphStyle(
            "EE6222Caption", parent=base["Normal"], fontName="Times-Roman",
            fontSize=8.3, leading=10, alignment=TA_LEFT,
            textColor=black, spaceBefore=1, spaceAfter=4,
        ),
        "table": ParagraphStyle(
            "EE6222Table", parent=base["Normal"], fontName="Helvetica",
            fontSize=8.0, leading=9.6, alignment=TA_LEFT, textColor=black,
        ),
        "table_header": ParagraphStyle(
            "EE6222TableHeader", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8.0, leading=9.6, alignment=TA_LEFT, textColor=black,
        ),
        "reference": ParagraphStyle(
            "EE6222Reference", parent=base["Normal"], fontName="Times-Roman",
            fontSize=8.2, leading=9.9, alignment=TA_LEFT,
            leftIndent=4 * mm, firstLineIndent=-4 * mm, spaceAfter=3,
            textColor=black,
        ),
        "code_header": ParagraphStyle(
            "EE6222CodeHeader", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=9.2, leading=11, spaceBefore=2, spaceAfter=3,
            textColor=black,
        ),
        "code": ParagraphStyle(
            "EE6222Code", fontName=MONO_FONT, fontSize=6.0, leading=7.2,
            leftIndent=0, rightIndent=0, spaceAfter=0,
        ),
        "appendix_intro": ParagraphStyle(
            "EE6222AppendixIntro", parent=base["BodyText"],
            fontName="Times-Roman", fontSize=8.8, leading=10.6,
            alignment=TA_LEFT, spaceAfter=4, textColor=black,
        ),
    }


def inline_markup(text: str) -> str:
    escaped = escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`",
                     r'<font name="Courier">\1</font>', escaped)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"\*(.+?)\*", r"<i>\1</i>", escaped)
    return escaped


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(inline_markup(text), style)


def split_markdown_table(lines: list[str], st: dict[str, ParagraphStyle]) -> Table:
    parsed: list[list[str]] = []
    for line in lines:
        row = line.strip().strip("|")
        cells = [cell.strip() for cell in row.split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        parsed.append(cells)
    table_data: list[list[Paragraph]] = []
    for row_number, row in enumerate(parsed):
        cell_style = st["table_header"] if row_number == 0 else st["table"]
        table_data.append([paragraph(cell, cell_style) for cell in row])
    widths_by_columns = {
        2: [87 * mm, 87 * mm],
        3: [42 * mm, 64 * mm, 68 * mm],
        4: [36 * mm, 54 * mm, 50 * mm, 34 * mm],
        5: [40 * mm, 24 * mm, 31 * mm, 42 * mm, 35 * mm],
    }
    column_count = len(table_data[0])
    if column_count not in widths_by_columns:
        raise ValueError(f"Unsupported Markdown table width: {column_count} columns")
    table = Table(table_data, colWidths=widths_by_columns[column_count],
                  repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#888888")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.55, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def figure_flowable(alt: str, relative_path: str,
                    st: dict[str, ParagraphStyle]) -> KeepTogether:
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Figure referenced by report is missing: {path}")
    width = 151 * mm
    height = 91 * mm if "lda_mean" in relative_path or "pca_mean" in relative_path else 84 * mm
    return KeepTogether(
        [Image(str(path), width=width, height=height, kind="proportional"),
         paragraph(alt, st["caption"])]
    )


def numbered_source(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(f"{line_number:4d} | {line}"
                     for line_number, line in enumerate(lines, 1))


def appendix_story(story: list[Any], st: dict[str, ParagraphStyle]) -> None:
    story.extend(
        [
            PageBreak(),
            paragraph("Appendix A. Complete Executable Program", st["section"]),
            paragraph(
                "This complete program loads Olivetti Faces through scikit-learn "
                "(using an existing cache or downloading it when absent), runs nested "
                "dimension selection and fixed-curve experiments, computes accuracy "
                "and error rates, aggregates outer-fold means and standard deviations, "
                "and writes tables, figures, a manifest, and validation results. "
                "The public repository provides this standalone script as its "
                "reproducible program.",
                st["appendix_intro"],
            ),
        ]
    )
    for index, relative in enumerate(APPENDIX_FILES):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Appendix source file is missing: {path}")
        if index:
            story.append(PageBreak())
        story.append(paragraph(relative, st["code_header"]))
        code = numbered_source(path)
        max_chars = max((len(line) for line in code.splitlines()), default=0)
        max_width = pdfmetrics.stringWidth(
            "M" * max_chars, MONO_FONT, st["code"].fontSize
        )
        if max_width > 490:
            raise ValueError(
                f"Appendix line in {relative} exceeds printable width: "
                f"{max_chars} chars, {max_width:.1f} pt"
            )
        story.append(Preformatted(code, st["code"]))


def parse_report(story: list[Any], st: dict[str, ParagraphStyle],
                 report_source: Path) -> None:
    lines = report_source.read_text(encoding="utf-8").splitlines()
    index = 0
    in_references = False
    in_title_block = False
    appendix_added = False
    image_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line == "<!-- PAGEBREAK -->":
            index += 1
            continue
        if line == "<!-- APPENDIX -->":
            appendix_story(story, st)
            appendix_added = True
            index += 1
            break
        if line.startswith("# "):
            story.append(paragraph(line[2:], st["title"]))
            in_title_block = True
            index += 1
            continue
        if line.startswith("## "):
            heading = line[3:]
            in_references = heading == "References"
            in_title_block = False
            story.append(paragraph(heading, st["section"]))
            index += 1
            continue
        if line.startswith("### "):
            story.append(paragraph(line[4:], st["subsection"]))
            index += 1
            continue
        image_match = image_pattern.fullmatch(line)
        if image_match:
            story.append(figure_flowable(image_match.group(1), image_match.group(2), st))
            index += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            story.append(split_markdown_table(table_lines, st))
            story.append(Spacer(1, 3))
            continue
        paragraph_lines = [line]
        index += 1
        while index < len(lines):
            next_line = lines[index].strip()
            if (
                not next_line
                or next_line.startswith("# ")
                or next_line.startswith("## ")
                or next_line.startswith("### ")
                or next_line.startswith("|")
                or next_line.startswith("![")
                or next_line.startswith("<!-- ")
            ):
                break
            paragraph_lines.append(next_line)
            index += 1
        text = " ".join(paragraph_lines)
        if in_title_block:
            style = st["author"]
        elif in_references:
            style = st["reference"]
        else:
            style = st["body"]
        story.append(paragraph(text, style))
    if not appendix_added:
        raise ValueError("Report source is missing the APPENDIX marker.")


class NumberedCanvas:
    def __new__(cls, *args: Any, **kwargs: Any):
        from reportlab.pdfgen.canvas import Canvas

        class _Canvas(Canvas):
            def __init__(self, *canvas_args: Any, **canvas_kwargs: Any):
                super().__init__(*canvas_args, **canvas_kwargs)
                self._saved_page_states: list[dict[str, Any]] = []

            def showPage(self) -> None:
                self._saved_page_states.append(dict(self.__dict__))
                self._startPage()

            def save(self) -> None:
                page_count = len(self._saved_page_states)
                for state in self._saved_page_states:
                    self.__dict__.update(state)
                    self.draw_footer(page_count)
                    Canvas.showPage(self)
                Canvas.save(self)

            def draw_footer(self, page_count: int) -> None:
                width, _ = A4
                self.saveState()
                self.setFont("Helvetica", 8)
                self.setFillColor(colors.black)
                self.drawCentredString(width / 2, 7 * mm,
                                      f"Page {self._pageNumber} of {page_count}")
                self.restoreState()

        return _Canvas(*args, **kwargs)


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def build(report_source: Path, pdf_path: Path, author: str) -> int:
    register_fonts()
    st = styles()
    if not report_source.is_file():
        raise FileNotFoundError(report_source)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Dimensionality Reduction for Face Classification Using PCA and LDA",
        author=author,
        subject="EE6222 Assignment 1",
        creator="ReportLab",
    )
    story: list[Any] = []
    parse_report(story, st, report_source)
    document.build(story, canvasmaker=NumberedCanvas)
    pages: int | None = None
    try:
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)], check=True, capture_output=True,
            text=True,
        )
        pages = next(
            int(line.split(":", 1)[1].strip())
            for line in result.stdout.splitlines()
            if line.startswith("Pages:")
        )
    except (OSError, subprocess.CalledProcessError, StopIteration, ValueError):
        pass
    print(f"PDF: {pdf_path}")
    print(f"Total pages: {pages if pages is not None else 'pdfinfo unavailable'}")
    print(f"Appendix source files: {len(APPENDIX_FILES)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the public EE6222 report PDF from its Markdown source."
    )
    parser.add_argument("--report-source", type=Path, default=DEFAULT_REPORT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_PDF_PATH)
    parser.add_argument("--author", default=DEFAULT_AUTHOR)
    args = parser.parse_args()
    return build(
        resolve_project_path(args.report_source),
        resolve_project_path(args.output),
        args.author,
    )


if __name__ == "__main__":
    raise SystemExit(main())
