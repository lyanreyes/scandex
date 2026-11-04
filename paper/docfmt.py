"""Formatting helpers that reproduce the Klarify IMRaD layout in python-docx:
single-column title block + abstract, then a two-column body."""
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

FONT = "Times New Roman"
BODY_PT = 11
COL_W = Inches(3.15)


class Paper:
    def __init__(self):
        self.doc = Document()
        s = self.doc.sections[0]
        s.page_width, s.page_height = Inches(8.5), Inches(11)
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(s, side, Inches(1))
        st = self.doc.styles["Normal"]
        st.font.name, st.font.size = FONT, Pt(BODY_PT)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        pf = st.paragraph_format
        pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(6), 1.0
        self.fig_n = 0
        self.tab_n = 0

    # ---------- primitives ----------
    def para(self, text="", align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None, bold=False, italic=False,
             indent=True, after=6, before=0, keep=False):
        p = self.doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_after = Pt(after)
        p.paragraph_format.space_before = Pt(before)
        if keep:
            p.paragraph_format.keep_with_next = True
        if indent and align == WD_ALIGN_PARAGRAPH.JUSTIFY:
            p.paragraph_format.first_line_indent = Inches(0.4)
        if text:
            self.runs(p, text, size=size, bold=bold, italic=italic)
        return p

    @staticmethod
    def runs(p, text, size=None, bold=False, italic=False):
        """**bold** and ~italic~ inline markup."""
        import re
        for part in re.split(r"(\*\*[^*]+\*\*|~[^~]+~)", text):
            if not part:
                continue
            b, i, t = bold, italic, part
            if part.startswith("**"):
                b, t = True, part[2:-2]
            elif part.startswith("~") and part.endswith("~") and len(part) > 2:
                i, t = True, part[1:-1]
            r = p.add_run(t)
            r.bold, r.italic = b, i
            if size:
                r.font.size = Pt(size)
        return p

    def title_block(self, title, authors, affiliation_lines):
        self.para(title, WD_ALIGN_PARAGRAPH.CENTER, size=14, bold=True, indent=False, after=8)
        self.para(authors, WD_ALIGN_PARAGRAPH.CENTER, size=11, indent=False, after=4)
        for line in affiliation_lines:
            self.para(line, WD_ALIGN_PARAGRAPH.CENTER, size=10, italic=True, indent=False, after=0)
        self.para("", after=6)

    def abstract(self, text, key_terms):
        self.para("ABSTRACT", WD_ALIGN_PARAGRAPH.CENTER, bold=True, indent=False, after=4)
        p = self.para(text, indent=False)
        p.paragraph_format.left_indent = p.paragraph_format.right_indent = Inches(0.3)
        p = self.para(f"**Key terms:** {key_terms}", WD_ALIGN_PARAGRAPH.LEFT, indent=False, after=10)
        p.paragraph_format.left_indent = Inches(0.3)

    def two_columns(self):
        sec = self.doc.add_section(WD_SECTION.CONTINUOUS)
        cols = sec._sectPr.xpath("./w:cols")
        el = cols[0] if cols else OxmlElement("w:cols")
        el.set(qn("w:num"), "2")
        el.set(qn("w:space"), "432")
        if not cols:
            sec._sectPr.append(el)

    def heading(self, roman, text):
        self.para(f"{roman}.  {text.upper()}", WD_ALIGN_PARAGRAPH.CENTER, bold=True, indent=False,
                  before=8, after=4, keep=True)

    def sub(self, text):
        self.para(text, WD_ALIGN_PARAGRAPH.LEFT, bold=True, indent=False, before=4, after=3, keep=True)

    def figure(self, path, caption, width=COL_W):
        self.fig_n += 1
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_after = Pt(2)
        p.add_run().add_picture(str(path), width=width)
        self.para(f"Figure {self.fig_n}. {caption}", WD_ALIGN_PARAGRAPH.CENTER, size=9.5,
                  italic=True, indent=False, after=8)
        return self.fig_n

    def table(self, caption, rows, widths=None, size=9, bold_last=False):
        self.tab_n += 1
        self.para(f"Table {self.tab_n}. {caption}", WD_ALIGN_PARAGRAPH.CENTER, size=9.5,
                  italic=True, indent=False, after=2, keep=True)
        t = self.doc.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        if not widths:
            widths = [int(COL_W / len(rows[0]))] * len(rows[0])
        layout = OxmlElement("w:tblLayout")
        layout.set(qn("w:type"), "fixed")
        t._tbl.tblPr.append(layout)
        for c, w in enumerate(widths):
            t.columns[c].width = w
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                cell = t.cell(r, c)
                cell.text = ""
                p = cell.paragraphs[0]
                p.paragraph_format.space_after = Pt(0)
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c == 0 else WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run(str(val))
                run.font.size = Pt(size)
                run.bold = r == 0 or (bold_last and r == len(rows) - 1)
                if r == 0:
                    sh = OxmlElement("w:shd")
                    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), "E8E2F7")
                    cell._tc.get_or_add_tcPr().append(sh)
                if widths:
                    cell.width = widths[c]
        self.para("", after=4)
        return self.tab_n

    def references(self, refs):
        for i, ref in enumerate(refs, 1):
            p = self.para("", WD_ALIGN_PARAGRAPH.LEFT, indent=False, after=4)
            p.paragraph_format.left_indent = Inches(0.35)
            p.paragraph_format.first_line_indent = Inches(-0.35)
            self.runs(p, f"[{i}]\t{ref}", size=10)

    def save(self, path):
        self.doc.save(path)
