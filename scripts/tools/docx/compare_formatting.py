"""
Temporary script to compare formatting of two DOCX files for IWA journal.
"""
import sys
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from collections import defaultdict, OrderedDict

AM = {
    WD_ALIGN_PARAGRAPH.LEFT: 'LEFT',
    WD_ALIGN_PARAGRAPH.CENTER: 'CENTER',
    WD_ALIGN_PARAGRAPH.RIGHT: 'RIGHT',
    WD_ALIGN_PARAGRAPH.JUSTIFY: 'JUSTIFY',
    None: 'None(inh)',
}

def ec(v):
    if v is None: return 'None'
    return f'{v/360000:.2f} cm'

def ps(v):
    if v is None: return 'None'
    return f'{v.pt:.1f} pt'

def cstr(c):
    if c is None: return 'None'
    try:
        if c.rgb is not None: return f'#{c.rgb}'
        if c.theme_color is not None: return f'Theme:{c.theme_color}'
    except: pass
    return 'None'

def als(a): return AM.get(a, str(a))
def ext_sec(doc):
    o = []
    for i, s in enumerate(doc.sections):
        d = OrderedDict()
        d['Section'] = i+1
        d['PageWidth'] = ec(s.page_width)
        d['PageHeight'] = ec(s.page_height)
        d['Orientation'] = str(s.orientation).split('.')[-1]
        d['TopMargin'] = ec(s.top_margin)
        d['BottomMargin'] = ec(s.bottom_margin)
        d['LeftMargin'] = ec(s.left_margin)
        d['RightMargin'] = ec(s.right_margin)
        d['HeaderDist'] = ec(s.header_distance)
        d['FooterDist'] = ec(s.footer_distance)
        d['Gutter'] = ec(s.gutter)
        o.append(d)
    return o

def ext_sty(st):
    d = OrderedDict()
    d['Name'] = st.name
    d['ID'] = st.style_id
    d['Type'] = str(st.type).split('.')[-1]
    d['Builtin'] = str(st.builtin)
    d['Base'] = st.base_style.name if st.base_style else 'None'
    f = st.font
    if f:
        d['Font'] = str(f.name) if f.name else 'None(inh)'
        d['Size'] = ps(f.size)
        d['Bold'] = str(f.bold) if f.bold is not None else 'None(inh)'
        d['Italic'] = str(f.italic) if f.italic is not None else 'None(inh)'
        d['Underline'] = str(f.underline) if f.underline is not None else 'None(inh)'
        d['Color'] = cstr(f.color)
        d['AllCaps'] = str(f.all_caps) if f.all_caps is not None else 'None(inh)'
    pf = getattr(st, 'paragraph_format', None)
    if pf:
        d['Align'] = als(pf.alignment)
        d['SpBefore'] = ps(pf.space_before)
        d['SpAfter'] = ps(pf.space_after)
        d['LineSp'] = str(pf.line_spacing) if pf.line_spacing else 'None(inh)'
        d['LineSpRule'] = str(pf.line_spacing_rule).split('.')[-1] if pf.line_spacing_rule else 'None(inh)'
        d['1stLineInd'] = ps(pf.first_line_indent)
        d['LeftInd'] = ps(pf.left_indent)
        d['RightInd'] = ps(pf.right_indent)
        d['KeepTog'] = str(pf.keep_together) if pf.keep_together is not None else 'None(inh)'
        d['KeepNext'] = str(pf.keep_with_next) if pf.keep_with_next is not None else 'None(inh)'
    return d

def ext_allsty(doc):
    o = OrderedDict()
    for s in doc.styles:
        try: o[s.name] = ext_sty(s)
        except Exception as e: o[s.name] = {'Err': str(e)}
    return o

def ext_used(doc):
    u = defaultdict(int)
    for p in doc.paragraphs: u[p.style.name if p.style else 'None'] += 1
    return dict(sorted(u.items(), key=lambda x: -x[1]))
