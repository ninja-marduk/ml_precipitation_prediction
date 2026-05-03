import sys, io, glob, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from docx import Document

def emu_pt(val):
    return round(val/12700, 1) if val else None

def get_all_runs(p):
    runs = []
    for r in p.runs:
        rf = r.font
        c = None
        try: c = str(rf.color.rgb)
        except: pass
        has_img = bool(r._element.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing") or
                       r._element.findall(".//{urn:schemas-microsoft-com:vml}imagedata"))
        runs.append({
            "text": r.text[:50] if r.text else "",
            "font": rf.name,
            "size": emu_pt(rf.size) if rf.size else None,
            "bold": rf.bold,
            "italic": rf.italic,
            "color": c,
            "has_img": has_img,
        })
    return runs

def get_para_fmt(p):
    pf = p.paragraph_format
    align = None
    try:
        a = pf.alignment
        align = str(a).split(".")[-1].split("(")[0].strip() if a else None
    except: pass
    return {
        "style": p.style.name if p.style else None,
        "align": align,
        "sb": emu_pt(pf.space_before) if pf.space_before else None,
        "sa": emu_pt(pf.space_after) if pf.space_after else None,
        "fi": emu_pt(pf.first_line_indent) if pf.first_line_indent else None,
        "li": emu_pt(pf.left_indent) if pf.left_indent else None,
        "ls": pf.line_spacing,
    }

def compare_element(label, orig_p, gen_p):
    print()
    print(f"--- {label} ---")
    ofmt = get_para_fmt(orig_p)
    gfmt = get_para_fmt(gen_p)
    oruns = get_all_runs(orig_p)
    gruns = get_all_runs(gen_p)
    gaps = []
    for key in ["align", "sb", "sa", "fi", "li", "ls"]:
        ov = ofmt[key]
        gv = gfmt[key]
        if ov == gv: continue
        if isinstance(ov, (int, float)) and isinstance(gv, (int, float)) and abs(ov - gv) < 1: continue
        if ov is None and gv == 0: continue
        if gv is None and ov == 0: continue
        gaps.append(f"  PARA {key}: orig={ov} gen={gv}")
    if oruns and gruns:
        o_r = oruns[0]
        g_r = gruns[0]
        for key in ["font", "size", "bold", "italic", "color"]:
            ov = o_r[key]
            gv = g_r[key]
            if ov == gv: continue
            if ov is None and gv is False: continue
            if gv is None and ov is False: continue
            if key == "color" and ov in (None, "None") and gv in (None, "None"): continue
            if key == "size" and ov is None and gv is not None:
                gaps.append(f"  RUN {key}: orig=inherit gen={gv}")
                continue
            if key == "bold" and ov is None and gv is True:
                gaps.append(f"  RUN {key}: orig=inherit gen={gv}")
                continue
            gaps.append(f"  RUN {key}: orig={ov} gen={gv}")
    o_imgs = sum(1 for r in oruns if r["has_img"])
    g_imgs = sum(1 for r in gruns if r["has_img"])
    if o_imgs != g_imgs:
        gaps.append(f"  IMAGES: orig={o_imgs} gen={g_imgs}")
    if len(oruns) != len(gruns):
        gaps.append(f"  RUN_COUNT: orig={len(oruns)} gen={len(gruns)}")
    o_text = orig_p.text.strip()[:100]
    g_text = gen_p.text.strip()[:100]
    if o_text != g_text:
        gaps.append("  TEXT DIFF:")
        gaps.append(f"    orig={o_text}")
        gaps.append(f"    gen ={g_text}")
    if gaps:
        skey = "style"
        print(f"  style: orig={ofmt[skey]} gen={gfmt[skey]}")
        for g in gaps:
            print(g)
    else:
        print("  MATCH OK")

orig = None
for f in glob.glob("docs/papers/1/Revis*/**/*v3.docx", recursive=True):
    orig = f
    break
print(f"Original file: {orig}")
doc_o = Document(orig)
doc_g = Document("docs/papers/1/latex/paper_revision4_formatted_v6.docx")
op = doc_o.paragraphs
gp = doc_g.paragraphs
print(f"Original paragraphs: {len(op)}, Generated paragraphs: {len(gp)}")

def find_para(paras, test_fn, start=0):
    for i in range(start, len(paras)):
        if test_fn(paras[i]):
            return i, paras[i]
    return None, None

_, o_title = find_para(op, lambda p: p.style and "Title" in p.style.name)
_, g_title = find_para(gp, lambda p: p.style and "Title" in p.style.name)
if o_title and g_title:
    compare_element("TITLE", o_title, g_title)

_, o_auth = find_para(op, lambda p: p.style and "Author" in p.style.name)
_, g_auth = find_para(gp, lambda p: p.style and "Author" in p.style.name)
if o_auth and g_auth:
    compare_element("AUTHORS", o_auth, g_auth)

o_auth_idx = next((i for i, p in enumerate(op) if p.style and "Author" in p.style.name), 0)
g_auth_idx = next((i for i, p in enumerate(gp) if p.style and "Author" in p.style.name), 0)

o_abs_idx, o_abs = find_para(op, lambda p: p.text.strip().upper() == "ABSTRACT")
g_abs_idx, g_abs = find_para(gp, lambda p: p.text.strip().upper() == "ABSTRACT")

ORCID_PAT = r"\d{4}-\d{4}-\d{4}-\d{4}"

if o_auth_idx and o_abs_idx:
    for i in range(o_auth_idx+1, o_abs_idx):
        p = op[i]
        txt = p.text.strip()
        if txt and "orcid" not in txt.lower() and not re.search(ORCID_PAT, txt) and not all(c=="_" for c in txt):
            for j in range(g_auth_idx+1, g_abs_idx if g_abs_idx else len(gp)):
                gpp = gp[j]
                gtxt = gpp.text.strip()
                if gtxt and "orcid" not in gtxt.lower() and not re.search(ORCID_PAT, gtxt) and not all(c=="_" for c in gtxt):
                    compare_element("AFFILIATIONS", p, gpp)
                    break
            break

_, o_orcid = find_para(op, lambda p: re.search(ORCID_PAT, p.text))
_, g_orcid = find_para(gp, lambda p: re.search(ORCID_PAT, p.text))
if o_orcid and g_orcid:
    compare_element("ORCID LINE", o_orcid, g_orcid)

if o_abs and g_abs:
    compare_element("ABSTRACT HEADING", o_abs, g_abs)

def find_body_after(paras, heading_idx):
    for i in range(heading_idx+1, min(heading_idx+5, len(paras))):
        if len(paras[i].text.strip()) > 50:
            return i, paras[i]
    return None, None

_, o_abody = find_body_after(op, o_abs_idx)
_, g_abody = find_body_after(gp, g_abs_idx)
if o_abody and g_abody:
    compare_element("ABSTRACT BODY", o_abody, g_abody)

_, o_kw = find_para(op, lambda p: p.text.strip().lower().startswith("key word"))
_, g_kw = find_para(gp, lambda p: p.text.strip().lower().startswith("key word"))
if o_kw and g_kw:
    compare_element("KEY WORDS", o_kw, g_kw)

o_hl_idx, o_hl = find_para(op, lambda p: p.text.strip().upper() == "HIGHLIGHTS")
g_hl_idx, g_hl = find_para(gp, lambda p: p.text.strip().upper() == "HIGHLIGHTS")
if o_hl and g_hl:
    compare_element("HIGHLIGHTS HEADING", o_hl, g_hl)

if o_hl_idx and g_hl_idx:
    _, o_hl1 = find_body_after(op, o_hl_idx)
    _, g_hl1 = find_body_after(gp, g_hl_idx)
    if not g_hl1:
        for i in range(g_hl_idx+1, min(g_hl_idx+5, len(gp))):
            if len(gp[i].text.strip()) > 20:
                g_hl1 = gp[i]
                break
    if o_hl1 and g_hl1:
        compare_element("HIGHLIGHT ITEM 1", o_hl1, g_hl1)

o_ga_idx, o_ga = find_para(op, lambda p: "GRAPHICAL" in p.text.strip().upper())
g_ga_idx, g_ga = find_para(gp, lambda p: "GRAPHICAL" in p.text.strip().upper())
if o_ga and g_ga:
    compare_element("GRAPHICAL ABSTRACT HEADING", o_ga, g_ga)

o_sec_idx, o_sec = find_para(op, lambda p: p.text.strip() and p.text.strip()[0].isdigit() and ("Head" in (p.style.name if p.style else "") or "Heading" in (p.style.name if p.style else "")))
g_sec_idx, g_sec = find_para(gp, lambda p: p.text.strip() and p.text.strip()[0].isdigit() and "Heading" in (p.style.name if p.style else ""))
if o_sec and g_sec:
    compare_element("FIRST NUMBERED HEADING", o_sec, g_sec)

_, o_div = find_para(op, lambda p: p.text.strip() and all(c=="_" for c in p.text.strip()) and len(p.text.strip()) >= 4)
_, g_div = find_para(gp, lambda p: p.text.strip() and all(c=="_" for c in p.text.strip()) and len(p.text.strip()) >= 4)
if o_div and g_div:
    compare_element("FIRST DIVIDER", o_div, g_div)
