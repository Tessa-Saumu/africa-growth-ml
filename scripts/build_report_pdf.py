"""Render reports/capstone_report.md to PDF with the project visual language.

AnalystLab deliverable #1 requires a PDF report. This environment has no
pandoc/LaTeX, so this script implements a small, deterministic Markdown
renderer (headings, paragraphs, lists, blockquotes, tables, fenced code,
embedded figures, links-as-text) on top of fpdf2 + DejaVu fonts. It reads
nothing but the Markdown file and the figures it references.

Usage:  python scripts/build_report_pdf.py [--md reports/capstone_report.md]
                                           [--pdf reports/capstone_report.pdf]
"""
import argparse
import logging
import re
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

from fpdf import FPDF


logger = logging.getLogger(__name__)

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
PROJECT_PRIMARY = (27, 79, 114)     # #1B4F72 (src.visualization palette)
PROJECT_SECONDARY = (46, 134, 193)  # #2E86C1
PROJECT_TEXT = (44, 62, 80)         # #2C3E50
PROJECT_LIGHT = (248, 249, 250)     # #F8F9FA

_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def strip_markdown_inline(text: str) -> str:
    """Remove inline markdown syntax (code spans, emphasis, links) for PDF text.

    Args:
        text: One logical line of markdown body text.

    Returns:
        Plain text with markers stripped (link labels kept).
    """
    text = _LINK.sub(r"\1", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _BOLD.sub(r"\1", text)
    text = _ITALIC.sub(r"\1", text)
    return text.strip()


class ReportPDF(FPDF):
    """FPDF subclass with project header/footer and section bookmarks."""

    def __init__(self) -> None:
        """Initialize A4 layout with margins."""
        super().__init__(format="A4")
        self.set_margins(left=18, top=18, right=18)
        self.set_auto_page_break(auto=True, margin=18)

    def header(self) -> None:
        """Render a thin project-colored band with document title."""
        if self.page_no() == 1:
            return
        self.set_fill_color(*PROJECT_LIGHT)
        self.rect(0, 0, 210, 10, style="F")
        self.set_text_color(*PROJECT_PRIMARY)
        self.set_font("DejaVu", size=8)
        self.set_y(3)
        self.cell(0, 5, "Africa Growth Explorer — Capstone Report", align="R")
        self.ln(8)

    def footer(self) -> None:
        """Render page number."""
        self.set_y(-14)
        self.set_text_color(127, 127, 127)
        self.set_font("DejaVu", size=8)
        self.cell(0, 10, f"{self.page_no()}", align="C")


def _find_font(name: str) -> Path:
    """Locate a DejaVu TTF on the system font dir or matplotlib's bundle.

    Args:
        name: Font file name, e.g. 'DejaVuSans.ttf'.

    Returns:
        Path to the font file.

    Raises:
        FileNotFoundError: If no candidate directory contains the font.
    """
    import matplotlib
    candidates = [
        FONT_DIR / name,
        Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name,
    ]
    for pth in candidates:
        if pth.exists():
            return pth
    raise FileNotFoundError(f"No {name} in {[str(c.parent) for c in candidates]}")


def register_fonts(pdf: ReportPDF) -> None:
    """Register DejaVu family so Unicode (–, ≥, é) renders.

    Args:
        pdf: Target ReportPDF instance.
    """
    pdf.add_font("DejaVu", "", _find_font("DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", _find_font("DejaVuSans-Bold.ttf"))
    try:
        pdf.add_font("DejaVu", "I", _find_font("DejaVuSans-Oblique.ttf"))
    except FileNotFoundError:
        pdf.add_font("DejaVu", "I", _find_font("DejaVuSans.ttf"))
    pdf.add_font("DejaVuMono", "", _find_font("DejaVuSansMono.ttf"))
    logger.info("Fonts registered")


def _emit_paragraph(pdf: ReportPDF, text: str, style: str = "", size: int = 10) -> None:
    """Write a justified paragraph in the given style.

    Args:
        pdf: Target document.
        text: Plain paragraph text.
        style: fpdf style flag ('', 'B', 'I').
        size: Font size in pt.
    """
    pdf.set_text_color(*PROJECT_TEXT)
    pdf.set_font("DejaVu", style, size)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5.0, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.4)


def _render_table(pdf: ReportPDF, rows: List[List[str]]) -> None:
    """Render a markdown table (first row = header) with proportional columns.

    Cells are pre-wrapped with textwrap (break_long_words) so unbreakable
    tokens (WDI codes) never overflow the column; row heights are exact.

    Args:
        pdf: Target document.
        rows: Cell strings per row; column count from the first row.
    """
    if not rows:
        return
    ncols = len(rows[0])
    total_w = pdf.epw
    font_size = 7.5
    char_w = font_size * 0.205  # approx pt->mm for DejaVu
    maxlens = [8] * ncols
    for row in rows:
        for j, cell in enumerate(row[:ncols]):
            maxlens[j] = max(maxlens[j], min(len(cell), 46))
    scale = total_w / (sum(maxlens) * char_w)
    widths = [max(10.0, int(maxlens[j] * char_w * scale) ) for j in range(ncols)]
    widths = [total_w * w / sum(widths) for w in widths]
    char_per_col = [max(4, int(w / char_w) - 1) for w in widths]

    pdf.set_draw_color(*PROJECT_SECONDARY)
    pdf.set_line_width(0.15)
    for i, row in enumerate(rows):
        row = list((row + [""] * ncols)[:ncols])
        wrapped = [textwrap.wrap(re.sub(r"\s+", " ", c), width=char_per_col[j],
                                 break_long_words=True, break_on_hyphens=False)
                   or [""] for j, c in enumerate(row)]
        n_lines = max(len(w) for w in wrapped)
        row_h = 3.6 * n_lines + 1.8
        if pdf.get_y() + row_h > pdf.page_break_trigger:
            pdf.add_page()
        y0, x0 = pdf.get_y(), pdf.l_margin
        if i == 0:
            pdf.set_fill_color(*PROJECT_PRIMARY)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("DejaVu", "B", font_size)
        else:
            pdf.set_text_color(*PROJECT_TEXT)
            pdf.set_font("DejaVu", size=font_size)
            pdf.set_fill_color(*(PROJECT_LIGHT if i % 2 else (255, 255, 255)))
        for j, lines in enumerate(wrapped):
            x = x0 + sum(widths[:j])
            # background, then border, then text (draw order matters)
            pdf.rect(x, y0, widths[j], row_h, style="F")
            pdf.rect(x, y0, widths[j], row_h, style="D")
            if i == 0:
                pdf.set_text_color(255, 255, 255)
            else:
                pdf.set_text_color(*PROJECT_TEXT)
            for k in range(n_lines):
                pdf.set_xy(x + 0.6, y0 + 0.9 + k * 3.6)
                pdf.cell(widths[j] - 1.2, 3.6, lines[k] if k < len(lines) else "")
        # restore fill color for the row (set per-column above)
        pdf.set_xy(x0, y0 + row_h)
    pdf.ln(2)


def _split_row(line: str) -> List[str]:
    """Split a markdown table row into stripped cells.

    Args:
        line: Raw table line (may start/end with '|').

    Returns:
        Cell strings.
    """
    parts = [c.strip() for c in line.strip().strip("|").split("|")]
    return parts


def _image_size(path: Path) -> Tuple[int, int]:
    """Return (width, height) in pixels for a PNG, for aspect-ratio scaling.

    Reads the IHDR chunk directly so the builder keeps its only hard
    dependency on fpdf2 rather than pulling in an imaging library.

    Args:
        path: Path to a PNG file.

    Returns:
        Tuple of (width, height) in pixels; (0, 0) if unreadable.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(24)
        if len(head) >= 24 and head[:8] == b"\x89PNG\r\n\x1a\n":
            return (int.from_bytes(head[16:20], "big"),
                    int.from_bytes(head[20:24], "big"))
    except OSError:
        logger.warning("Could not read image dimensions: %s", path)
    return (0, 0)


def _caption_height(lines: List[str], idx: int, pdf: "ReportPDF") -> float:
    """Estimate the height of a bold-lead caption paragraph following a figure.

    Used to decide whether a figure and its caption fit on the current page.
    Overestimates slightly, which is the safe direction: a spurious page break
    is cosmetic, a caption orphaned from its figure is not.

    Args:
        lines: All document lines.
        idx: Index of the line just after the image line.
        pdf: The PDF instance, for effective page width.

    Returns:
        Estimated caption height in millimetres (0.0 if no caption follows).
    """
    j = idx
    while j < len(lines) and not lines[j].strip():
        j += 1
    if j >= len(lines):
        return 0.0
    text = lines[j].strip()
    if not text.startswith("**Figure"):
        return 0.0
    chars_per_line = max(int(pdf.epw / 1.85), 1)
    return (len(text) / chars_per_line + 1) * 4.6 + 4.0


def render_markdown(pdf: ReportPDF, md_text: str, base_dir: Path) -> None:
    """Render markdown source into the pdf document object.

    Args:
        pdf: Target ReportPDF (fonts already registered).
        md_text: Full markdown source.
        base_dir: Directory resolving relative image paths.
    """
    lines = md_text.splitlines()
    i, n = 0, len(lines)
    in_code = False
    code_buf: List[str] = []
    while i < n:
        raw = lines[i]
        line = raw.rstrip()

        if line.strip().startswith("```"):
            if in_code:
                pdf.set_fill_color(*PROJECT_LIGHT)
                pdf.set_text_color(*PROJECT_TEXT)
                pdf.set_font("DejaVuMono", size=8)
                for cl in code_buf:
                    pdf.multi_cell(0, 4.2, cl, fill=True, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(1.6)
                code_buf = []
            in_code = not in_code
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # table detection
        if line.lstrip().startswith("|") and i + 1 < n and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            rows = [_split_row(line)]
            i += 2
            while i < n and lines[i].lstrip().startswith("|"):
                rows.append([strip_markdown_inline(c) for c in _split_row(lines[i])])
                i += 1
            _render_table(pdf, rows)
            continue

        # image-only line
        m_img = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", line.strip())
        if m_img:
            img_path = (base_dir / m_img.group(2)).resolve()
            if img_path.exists():
                # Keep the figure and its caption on one page. Scale the image
                # to the space actually left, and if the figure plus a caption
                # band cannot fit, break the page first rather than orphaning
                # the caption (a caption split across pages is unreadable).
                render_w = min(pdf.epw, 170)
                iw, ih = _image_size(img_path)
                render_h = render_w * ih / iw if iw else 0.0
                caption_band = _caption_height(lines, i + 1, pdf)
                avail = pdf.h - pdf.b_margin - pdf.get_y()
                if render_h + caption_band > avail:
                    max_h = pdf.h - pdf.b_margin - pdf.t_margin - caption_band
                    if render_h > max_h and max_h > 0:
                        # Too tall for any page at full width: scale to fit.
                        render_w = render_w * max_h / render_h
                        render_h = max_h
                    if render_h + caption_band > avail:
                        pdf.add_page()
                x = pdf.l_margin + (pdf.epw - render_w) / 2
                pdf.image(str(img_path), x=x, w=render_w)
                pdf.ln(2)
            else:
                logger.warning("Figure not found, skipping: %s", img_path)
            i += 1
            continue

        stripped = line.strip()
        if not stripped:
            pdf.ln(1.6)
        elif stripped == "---":
            pdf.ln(1)
            pdf.set_draw_color(*PROJECT_SECONDARY)
            pdf.set_line_width(0.3)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(3)
        elif stripped.startswith("####"):
            _emit_paragraph(pdf, strip_markdown_inline(stripped.lstrip("# ")), "B", 10.5)
        elif stripped.startswith("###"):
            pdf.ln(1)
            _emit_paragraph(pdf, strip_markdown_inline(stripped.lstrip("# ")), "B", 12)
        elif stripped.startswith("##"):
            pdf.ln(2)
            _emit_paragraph(pdf, strip_markdown_inline(stripped.lstrip("# ")), "B", 14.5)
        elif stripped.startswith("# "):
            _emit_paragraph(pdf, strip_markdown_inline(stripped[2:]), "B", 18)
        elif stripped.startswith(">"):
            pdf.set_text_color(*PROJECT_PRIMARY)
            pdf.set_font("DejaVu", "I", 10)
            quote = [strip_markdown_inline(l.lstrip("> ").rstrip())
                     for l in lines[i:i + 1]]
            # merge consecutive quote lines
            j = i + 1
            while j < n and lines[j].lstrip().startswith(">"):
                quote.append(strip_markdown_inline(lines[j].lstrip("> ").rstrip()))
                j += 1
            pdf.set_fill_color(*PROJECT_LIGHT)
            pdf.multi_cell(0, 5.0, " ".join(quote), fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1.6)
            i = j
            continue
        elif re.match(r"^[-*] ", stripped):
            pdf.set_text_color(*PROJECT_TEXT)
            pdf.set_font("DejaVu", size=10)
            pdf.multi_cell(0, 5.0, "•  " + strip_markdown_inline(stripped[2:]), new_x="LMARGIN", new_y="NEXT")
        elif re.match(r"^\d+\. ", stripped):
            num, rest = stripped.split(". ", 1)
            pdf.set_text_color(*PROJECT_TEXT)
            pdf.set_font("DejaVu", size=10)
            pdf.multi_cell(0, 5.0, f"{num}.  " + strip_markdown_inline(rest), new_x="LMARGIN", new_y="NEXT")
        else:
            # merge wrapped paragraph lines
            para = [stripped]
            j = i + 1
            while j < n and lines[j].strip() and not re.match(
                    r"^\s*([#>*-]|\||```|\d+\. |!\[)", lines[j]):
                para.append(lines[j].strip())
                j += 1
            _emit_paragraph(pdf, strip_markdown_inline(" ".join(para)))
            i = j
            continue
        i += 1
    if code_buf:
        pdf.set_font("DejaVuMono", size=8)
        for cl in code_buf:
            pdf.multi_cell(0, 4.2, cl, fill=True, new_x="LMARGIN", new_y="NEXT")


def main(md_path: Optional[Path] = None, pdf_path: Optional[Path] = None) -> Path:
    """Build the PDF report.

    Args:
        md_path: Source markdown (default reports/capstone_report.md).
        pdf_path: Output PDF (default reports/capstone_report.pdf).

    Returns:
        Path of the written PDF.

    Raises:
        FileNotFoundError: If the markdown source is missing.
    """
    md_path = Path(md_path) if md_path else Path("reports/capstone_report.md")
    pdf_path = Path(pdf_path) if pdf_path else md_path.with_suffix(".pdf")
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown source not found: {md_path}")

    pdf = ReportPDF()
    register_fonts(pdf)
    pdf.add_page()
    render_markdown(pdf, md_path.read_text(encoding="utf-8"), md_path.parent)
    pdf.output(str(pdf_path))
    logger.info("PDF written: %s (%.1f KB, %d pages)",
                pdf_path, pdf_path.stat().st_size / 1024, pdf.page_no())
    return pdf_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Render the capstone report to PDF")
    ap.add_argument("--md", type=Path, default=Path("reports/capstone_report.md"))
    ap.add_argument("--pdf", type=Path, default=None)
    args = ap.parse_args()
    main(md_path=args.md, pdf_path=args.pdf)
