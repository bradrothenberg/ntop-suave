"""House style for engineering-report PDFs (reportlab).

nTop design system + typography override:
  - Headings / title : Oswald SemiBold
  - Body text        : IBM Plex Sans
  - Image captions    : IBM Plex Mono Light
  - Code / commands  : IBM Plex Mono
Palette follows the nTop design skill (white ground, #cccccc rules,
black text, #666 muted, #1565c0 accent, greyscale table headers).

Usage in a make_report.py:

    from report_style import (make_doc, H1, H2, BODY, CAP, SUB, MONO, TITLE,
                              styled_table, fig_single, fig_pair, NT)
    doc, story = make_doc(OUT_PATH, title="...", author="...",
                          subtitle="...", footer="nTop | Fun3D")
    story.append(Paragraph("...", TITLE))
    ...
    doc.build(story)

Run with: uv run --with reportlab --with pillow python make_report.py

Fonts live in ../assets/fonts (TTF). If they are missing the module falls
back to Helvetica/Courier so a report still builds.
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Table,
                                TableStyle, Image, KeepTogether,
                                CondPageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from PIL import Image as PILImage

# --- nTop palette ------------------------------------------------------------
class NT:
    black   = colors.HexColor("#000000")
    text    = colors.HexColor("#1a1a1a")
    muted   = colors.HexColor("#666666")
    faint   = colors.HexColor("#999999")
    rule    = colors.HexColor("#cccccc")
    rule_lt = colors.HexColor("#e0e0e0")
    bg_hdr  = colors.HexColor("#f5f5f5")
    bg_sub  = colors.HexColor("#f9f9f9")
    accent  = colors.HexColor("#1565c0")
    good    = colors.HexColor("#1b5e20")
    warn    = colors.HexColor("#f57f17")
    error   = colors.HexColor("#b71c1c")
    white   = colors.white

# --- font registration -------------------------------------------------------
_FONTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "assets", "fonts")

def _try(name, filename):
    p = os.path.join(_FONTDIR, filename)
    if os.path.exists(p):
        pdfmetrics.registerFont(TTFont(name, p))
        return True
    return False

_HAVE_FONTS = _try("PlexSans",      "IBMPlexSans-Regular.ttf")
_try("PlexSans-SemiBold", "IBMPlexSans-SemiBold.ttf")
_try("PlexSans-Bold",     "IBMPlexSans-Bold.ttf")
_try("PlexSans-Italic",   "IBMPlexSans-Italic.ttf")
_try("PlexMonoLight",     "IBMPlexMono-Light.ttf")
_try("PlexMono",          "IBMPlexMono-Regular.ttf")
_HAVE_OSWALD = _try("Oswald-SemiBold", "Oswald-SemiBold.ttf")

if _HAVE_FONTS:
    # <b>/<i> markup inside IBM Plex Sans body text
    registerFontFamily("PlexSans", normal="PlexSans", bold="PlexSans-Bold",
                       italic="PlexSans-Italic", boldItalic="PlexSans-Bold")
    F_BODY, F_BOLD, F_SEMI = "PlexSans", "PlexSans-Bold", "PlexSans-SemiBold"
    F_CAP  = "PlexMonoLight"
    F_MONO = "PlexMono"
else:  # graceful fallback
    F_BODY = F_BOLD = F_SEMI = "Helvetica"
    F_CAP = F_MONO = "Courier"

F_HEAD = "Oswald-SemiBold" if _HAVE_OSWALD else F_SEMI

_styles = getSampleStyleSheet()

# Title + headings: Oswald SemiBold, tight tracking like the deck
TITLE = ParagraphStyle("Titlex", parent=_styles["Title"], fontName=F_HEAD,
                       fontSize=24, leading=27, alignment=TA_LEFT,
                       textColor=NT.black, spaceAfter=2)
SUBTITLE = ParagraphStyle("Subtitlex", parent=_styles["Normal"],
                          fontName=F_CAP, fontSize=10.5, leading=14,
                          textColor=NT.muted, spaceBefore=2, spaceAfter=2)
H1 = ParagraphStyle("H1x", parent=_styles["Heading1"], fontName=F_HEAD,
                    fontSize=15, leading=18, textColor=NT.black,
                    spaceBefore=15, spaceAfter=6, keepWithNext=1)
H2 = ParagraphStyle("H2x", parent=_styles["Heading2"], fontName=F_HEAD,
                    fontSize=12, leading=15, textColor=NT.accent,
                    spaceBefore=10, spaceAfter=4, keepWithNext=1)
BODY = ParagraphStyle("Bodyx", parent=_styles["Normal"], fontName=F_BODY,
                      fontSize=9.5, leading=14, alignment=TA_JUSTIFY,
                      textColor=NT.text, spaceAfter=6)
ABSTRACT = ParagraphStyle("Abstractx", parent=BODY, fontSize=10,
                          leading=14.5, leftIndent=10, rightIndent=10,
                          spaceBefore=4, spaceAfter=10, borderColor=NT.rule)
BULLET = ParagraphStyle("Bulletx", parent=BODY, leftIndent=16,
                        bulletIndent=4, spaceAfter=3, alignment=TA_LEFT)
# Captions: IBM Plex Mono Light
CAP = ParagraphStyle("Capx", parent=_styles["Normal"], fontName=F_CAP,
                     fontSize=8, leading=11, alignment=TA_LEFT,
                     textColor=NT.muted, spaceBefore=4, spaceAfter=14)
SUB = ParagraphStyle("Subx", parent=_styles["Normal"], fontName=F_CAP,
                     fontSize=8, leading=10.5, alignment=TA_CENTER,
                     textColor=NT.muted)
MONO = ParagraphStyle("Monox", parent=_styles["Code"], fontName=F_MONO,
                      fontSize=8, leading=11.5, leftIndent=12, spaceBefore=4,
                      spaceAfter=8, textColor=NT.text,
                      backColor=NT.bg_sub, borderPadding=6)
# Table body cells: wraps long text inside reportlab Tables (raw strings
# do not wrap). Use cell_table() for any table whose cells may be long.
CELL = ParagraphStyle("Cellx", parent=_styles["Normal"], fontName=F_BODY,
                      fontSize=8.5, leading=11, alignment=TA_LEFT,
                      textColor=NT.text)

# --- table style (nTop greyscale header) ------------------------------------
TBL_STYLE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), NT.bg_hdr),
    ("TEXTCOLOR", (0, 0), (-1, 0), NT.black),
    ("LINEBELOW", (0, 0), (-1, 0), 1.2, NT.black),
    ("FONTNAME", (0, 0), (-1, 0), F_SEMI),
    ("FONTNAME", (0, 1), (-1, -1), F_BODY),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("TEXTCOLOR", (0, 1), (-1, -1), NT.text),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [NT.white, NT.bg_sub]),
    ("GRID", (0, 0), (-1, -1), 0.5, NT.rule),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
])


def _page_furniture(footer_text):
    """Return an onPage callback: hairline + running footer, nTop style."""
    def draw(canvas, doc):
        canvas.saveState()
        w, h = letter
        y = 0.55 * inch
        canvas.setStrokeColor(NT.rule)
        canvas.setLineWidth(0.6)
        canvas.line(0.85 * inch, y + 8, w - 0.85 * inch, y + 8)
        canvas.setFont(F_CAP, 7.5)
        canvas.setFillColor(NT.muted)
        canvas.drawString(0.85 * inch, y - 2, footer_text)
        canvas.drawRightString(w - 0.85 * inch, y - 2,
                               "p. %d" % canvas.getPageNumber())
        canvas.restoreState()
    return draw


def make_doc(out_path, title, author, subtitle=None,
             footer="nTop  |  engineering report"):
    doc = SimpleDocTemplate(out_path, pagesize=letter,
                            leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                            topMargin=0.8 * inch, bottomMargin=0.9 * inch,
                            title=title, author=author)
    doc._nt_onpage = _page_furniture(footer)
    return doc, []


def build(doc, story):
    """Build with the running footer on every page."""
    cb = getattr(doc, "_nt_onpage", None)
    if cb is None:
        doc.build(story)
    else:
        doc.build(story, onFirstPage=cb, onLaterPages=cb)


def sect(story, text, style=H1, min_space_in=1.5):
    """Append a section heading that never sits orphaned at a page bottom:
    requires min_space_in of room left on the page, else breaks first.
    (keepWithNext on the style alone is unreliable with KeepTogether
    figures, so use this instead of story.append(Paragraph(text, H1)).)"""
    story.append(CondPageBreak(min_space_in * inch))
    story.append(Paragraph(text, style))


def hrule(width_in=6.8, thickness=1.2, color=NT.black, space_after=8):
    """A horizontal rule as a thin table (heading underline, nTop style)."""
    t = Table([[""]], colWidths=[width_in * inch], rowHeights=[0.5])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), thickness, color)]))
    t.spaceAfter = space_after
    return t


def styled_table(rows, col_widths_in, align=None):
    t = Table(rows, colWidths=[w * inch for w in col_widths_in],
              repeatRows=1)
    t.setStyle(TBL_STYLE)
    if align:  # optional per-column alignment list, e.g. ["LEFT","RIGHT",...]
        for i, a in enumerate(align):
            t.setStyle(TableStyle([("ALIGN", (i, 0), (i, -1), a)]))
    return t


def cell_table(rows, col_widths_in, align=None):
    """styled_table with every non-header cell wrapped in a Paragraph so
    long cell text wraps instead of overflowing the column."""
    wrapped = [rows[0]] + [[Paragraph(str(c), CELL) for c in r]
                           for r in rows[1:]]
    return styled_table(wrapped, col_widths_in, align)


def fig_single(path, num, caption, width_in=6.6, max_h_in=4.6):
    pw, ph = PILImage.open(path).size
    w = width_in * inch
    h = w * ph / pw
    if h > max_h_in * inch:
        h = max_h_in * inch
        w = h * pw / ph
    return KeepTogether([Image(path, width=w, height=h),
                         Paragraph("Figure %d. %s" % (num, caption), CAP)])


def fig_pair(path_a, path_b, num, caption,
             lab_a="(a) baseline", lab_b="(b) result", width_in=3.30,
             max_h_in=3.2):
    """Side-by-side figure pair. Per-image aspect is always preserved; if
    either image would exceed max_h_in the row is scaled down."""
    cell_w = width_in * inch
    sizes = []
    for p in (path_a, path_b):
        pw, ph = PILImage.open(p).size
        sizes.append((pw, ph))
    h_max = max(cell_w * ph / pw for pw, ph in sizes)
    if h_max > max_h_in * inch:
        scale = max_h_in * inch / h_max
    else:
        scale = 1.0

    def _img(p, pw, ph):
        w = cell_w * scale
        h = w * ph / pw
        return Image(p, width=w, height=h)

    (paw, pah), (pbw, pbh) = sizes
    grid = Table([[_img(path_a, paw, pah), _img(path_b, pbw, pbh)],
                  [Paragraph(lab_a, SUB), Paragraph(lab_b, SUB)]],
                 colWidths=[cell_w + 4, cell_w + 4])
    grid.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))
    return KeepTogether([grid,
                         Paragraph("Figure %d. %s" % (num, caption), CAP)])
