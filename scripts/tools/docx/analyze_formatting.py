import sys, io, glob, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from docx import Document
from docx.oxml.ns import qn
def emu_pt(val):
    return round(val/12700, 1) if val else None
def fmt_align(a):
    if a is None: return "inherit"
    try: return str(a).split(".")[-1].split("(")[0].strip()
    except: return str(a)
orig = None
for f in glob.glob("docs/papers/1/Revis*/**/*v3.docx", recursive=True):
    orig = f
    break
print("Original: " + str(orig))
doc = Document(orig)
paras = doc.paragraphs
print()
print("=== FIRST PARAGRAPH AFTER HEADING ===")
for i, p2 in enumerate(paras):
    sn = p2.style.name if p2.style else ""
    if sn in ("Head1","Heading 1") and p2.text.strip() and p2.text.strip()[0].isdigit():
        for j in range(i+1, min(i+5, len(paras))):
            pp = paras[j]
            ps = pp.style.name if pp.style else ""
            txt = pp.text.strip()
            if txt and len(txt) > 30:
                fi = pp.paragraph_format.first_line_indent
                fi_pt = emu_pt(fi) if fi else None
                pPr = pp._element.find(qn("w:pPr"))
                xml_ind = {}
                if pPr is not None:
                    ind = pPr.find(qn("w:ind"))
                    if ind is not None:
                        for attr in ["firstLine","hanging","left","right"]:
                            v = ind.get(qn("w:"+attr))
                            if v: xml_ind[attr] = v
                print("  After [{}] ".format(i) + repr(paras[i].text[:40]))
                print("    [{}] style={}, fi={}pt, xml_ind={}".format(j, ps, fi_pt, xml_ind))
                print("    text=" + repr(txt[:60]))
                break
print()
print("=== CAPTION FORMATTING ===")
for i, p2 in enumerate(paras):
    txt = p2.text.strip()
    lower = txt[:30].lower()
    if ("figure" in lower or "table" in lower) and "|" in txt[:40]:
        sn = p2.style.name if p2.style else ""
        for ri2, r in enumerate(p2.runs[:4]):
            rf = r.font
            color = None
            try: color = rf.color.rgb
            except: pass
            tc = None
            rPr = r._element.find(qn("w:rPr"))
            if rPr is not None:
                c_el = rPr.find(qn("w:color"))
                if c_el is not None:
                    tc = c_el.get(qn("w:themeColor"))
            if ri2 == 0:
                align = fmt_align(p2.paragraph_format.alignment)
                print("  [{}] style={} align={}".format(i, sn, align))
                print("    text=" + repr(txt[:80]))
            print("    run[{}]: text={} font={} sz={}pt b={} c={} theme={}".format(ri2, repr(r.text[:30]), rf.name, emu_pt(rf.size) if rf.size else None, rf.bold, str(color) if color else None, tc))
print()
print("=== TABLE FORMATTING ===")
for ti, table in enumerate(doc.tables):
    print("  Table {}: {} rows x {} cols".format(ti, len(table.rows), len(table.columns)))
    tbl = table._tbl
    tblPr = tbl.find(qn("w:tblPr"))
    if tblPr is not None:
        jc = tblPr.find(qn("w:jc"))
        if jc is not None:
            print("    table align: {}".format(jc.get(qn("w:val"))))
        tw = tblPr.find(qn("w:tblW"))
        if tw is not None:
            print("    table width: {} type={}".format(tw.get(qn("w:w")), tw.get(qn("w:type"))))
        ts = tblPr.find(qn("w:tblStyle"))
        if ts is not None:
            print("    table style: {}".format(ts.get(qn("w:val"))))
    if table.rows:
        row0 = table.rows[0]
        for ci, cell in enumerate(row0.cells[:3]):
            for cp in cell.paragraphs[:1]:
                rf_info = []
                for r in cp.runs[:2]:
                    rf = r.font
                    rf_info.append("font={} sz={} b={}".format(rf.name, emu_pt(rf.size) if rf.size else None, rf.bold))
                align = fmt_align(cp.paragraph_format.alignment)
                print("    cell[0,{}]: align={} text={} runs={}".format(ci, align, repr(cp.text[:30]), rf_info))
    if ti >= 3:
        print("  ... (showing first 4 tables)")
        break
print()
print("=== EQUATION PARAGRAPHS ===")
for i, p2 in enumerate(paras):
    sn = p2.style.name if p2.style else ""
    txt = p2.text.strip()
    elem = p2._element
    ns = "http://schemas.openxmlformats.org/officeDocument/2006/math"
    oMath = elem.findall(".//{" + ns + "}oMath")
    oMathPara = elem.findall(".//{" + ns + "}oMathPara")
    if oMath or oMathPara or (sn and "Ecuacion" in sn) or (sn and "Equation" in sn):
        align = fmt_align(p2.paragraph_format.alignment)
        pPr = elem.find(qn("w:pPr"))
        tabs_info = []
        if pPr is not None:
            tabs = pPr.find(qn("w:tabs"))
            if tabs:
                for tab in tabs.findall(qn("w:tab")):
                    tabs_info.append("val={} pos={}".format(tab.get(qn("w:val")), tab.get(qn("w:pos"))))
        print("  [{}] style={} align={} oMath={} oMathPara={}".format(i, sn, align, len(oMath), len(oMathPara)))
        print("    text=" + repr(txt[:80]))
        if tabs_info:
            print("    tabs: {}".format(tabs_info))
        if re.search(r"\(\d+\)", txt):
            print("    has equation number")
print()
print("=== REFERENCES SECTION ===")
in_refs = False
ref_count = 0
for i, p2 in enumerate(paras):
    sn = p2.style.name if p2.style else ""
    txt = p2.text.strip()
    upper = txt.upper()
    if "REFERENCE" in upper and ("Head" in sn or "Heading" in sn):
        print("  [{}] REFERENCES heading: style={}".format(i, sn))
        for r in p2.runs:
            if r.text.strip():
                rf_i = r.font
                c = None
                try: c = rf_i.color.rgb
                except: pass
                print("    font={} sz={} b={} c={}".format(rf_i.name, emu_pt(rf_i.size) if rf_i.size else None, rf_i.bold, c))
                break
        in_refs = True
        continue
    if in_refs and txt:
        if ref_count < 3:
            align = fmt_align(p2.paragraph_format.alignment)
            sa = emu_pt(p2.paragraph_format.space_after) if p2.paragraph_format.space_after else None
            sb = emu_pt(p2.paragraph_format.space_before) if p2.paragraph_format.space_before else None
            fi = emu_pt(p2.paragraph_format.first_line_indent) if p2.paragraph_format.first_line_indent else None
            ls = p2.paragraph_format.line_spacing
            for r in p2.runs[:1]:
                rf2 = r.font
                c2 = None
                try: c2 = rf2.color.rgb
                except: pass
                print("  [{}] style={} font={} sz={} b={} c={}".format(i, sn, rf2.name, emu_pt(rf2.size) if rf2.size else None, rf2.bold, c2))
                print("    align={} sa={} sb={} fi={} ls={}".format(align, sa, sb, fi, ls))
                print("    text=" + repr(txt[:80]))
        ref_count += 1
if in_refs:
    print("  Total references: {}".format(ref_count))
print()
print("=== HIGHLIGHT ITEMS ===")
in_hl = False
for i, p2 in enumerate(paras):
    sn = p2.style.name if p2.style else ""
    txt = p2.text.strip()
    if ("Head" in sn or "Heading" in sn) and txt.upper() == "HIGHLIGHTS":
        in_hl = True
        continue
    if in_hl:
        if ("Head" in sn or "Heading" in sn):
            break
        if txt:
            for r in p2.runs[:3]:
                rf = r.font
                print("  [{}] run: {} font={} sz={} b={}".format(i, repr(r.text[:40]), rf.name, emu_pt(rf.size) if rf.size else None, rf.bold))
            pf = p2.paragraph_format
            li = emu_pt(pf.left_indent) if pf.left_indent else None
            fi = emu_pt(pf.first_line_indent) if pf.first_line_indent else None
            print("    style={} left_indent={} first_indent={}".format(sn, li, fi))