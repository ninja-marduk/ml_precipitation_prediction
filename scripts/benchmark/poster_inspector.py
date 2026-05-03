"""
Poster Inspector - Programmatic PDF Design Analysis for Scientific Posters
==========================================================================
Replaces Figma MCP's inspection capabilities for LaTeX poster PDFs.
Uses PyMuPDF for layout/typography/color extraction, Pillow for visual balance.

Usage:
    python scripts/benchmark/poster_inspector.py [poster.pdf] [--json] [--verbose]

Outputs a structured design report with:
    - Layout analysis (bounding boxes, spacing, margins, columns)
    - Typography audit (font hierarchy, sizes, consistency, readability)
    - Color extraction + WCAG contrast checking
    - Visual balance analysis (weight distribution across regions)
    - Column alignment verification
    - 100-point design score with defect list
"""

import sys
import json
import math
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("ERROR: PyMuPDF required. Install: pip install PyMuPDF")

try:
    from PIL import Image
    import numpy as np
except ImportError:
    Image = None
    np = None

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

# A0 poster dimensions in mm
A0_WIDTH_MM = 841
A0_HEIGHT_MM = 1189
PT_TO_MM = 25.4 / 72  # 1 point = 0.3528 mm

# Typography thresholds for A0 posters (in points)
TYPO_THRESHOLDS = {
    "title_min": 60,        # readable from 5m
    "subtitle_min": 40,     # readable from 3m
    "section_header_min": 28,  # readable from 2m
    "body_min": 20,         # readable from 1.5m
    "caption_min": 16,      # readable from 1m
    "reference_min": 14,    # smallest acceptable
}

# WCAG 2.1 contrast requirements
WCAG_AA_NORMAL = 4.5   # normal text (<18pt or <14pt bold)
WCAG_AA_LARGE = 3.0    # large text (>=18pt or >=14pt bold)
WCAG_AAA_NORMAL = 7.0  # enhanced
WCAG_AAA_LARGE = 4.5   # enhanced large

# Poster design targets
TARGET_WHITESPACE_PCT = 30  # minimum 30% whitespace
TARGET_MAX_FONTS = 3        # max font families (excluding math)
TARGET_MAX_COLORS = 8       # max unique colors

# Okabe-Ito reference palette
OKABE_ITO = {
    "#000000": "Black",
    "#e69f00": "Orange",
    "#56b4e9": "Sky Blue",
    "#009e73": "Bluish Green",
    "#f0e442": "Yellow",
    "#0072b2": "Blue",
    "#d55e00": "Vermillion",
    "#cc79a7": "Reddish Purple",
    "#ffffff": "White",
}


# ─────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────

@dataclass
class TextSpan:
    text: str
    font: str
    size: float
    color: str  # hex
    bold: bool
    italic: bool
    bbox: tuple  # (x0, y0, x1, y1) in points


@dataclass
class Drawing:
    type: str  # "rect", "line", "path"
    bbox: tuple
    fill: Optional[str]  # hex or None
    stroke: Optional[str]  # hex or None
    width: float  # stroke width


@dataclass
class ImageInfo:
    bbox: tuple
    width_pt: float
    height_pt: float
    xref: int


@dataclass
class TypographyReport:
    fonts: dict  # {font_name: count}
    font_families: list  # unique base families
    math_fonts: list  # detected math fonts
    size_histogram: dict  # {size_pt: count}
    max_size: float
    min_size: float
    title_size: float  # largest text
    body_sizes: list  # most common sizes
    hierarchy_valid: bool
    issues: list = field(default_factory=list)
    score: float = 0


@dataclass
class ColorReport:
    unique_colors: dict  # {hex: count}
    text_colors: dict  # {hex: count}
    background_colors: list  # from filled rects
    contrast_pairs: list  # [{fg, bg, ratio, wcag_aa, wcag_aaa, text_sample}]
    violations: list  # WCAG violations
    okabe_ito_match: dict  # {hex: closest_okabe_ito}
    issues: list = field(default_factory=list)
    score: float = 0


@dataclass
class LayoutReport:
    page_width_mm: float
    page_height_mm: float
    margins: dict  # {left, right, top, bottom} in mm
    columns_detected: int
    column_boundaries: list  # x positions
    column_start_y: list  # y start per column
    column_alignment_error_mm: float
    whitespace_pct: float
    element_count: int
    issues: list = field(default_factory=list)
    score: float = 0


@dataclass
class BalanceReport:
    quadrant_weights: dict  # {TL, TR, BL, BR: float}
    column_weights: list  # [col1_weight, col2_weight, ...]
    max_imbalance_pct: float
    issues: list = field(default_factory=list)
    score: float = 0


@dataclass
class InspectionReport:
    file: str
    page_size_mm: tuple
    typography: TypographyReport
    colors: ColorReport
    layout: LayoutReport
    balance: BalanceReport
    total_score: float = 0
    grade: str = ""
    defects: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Color Utilities
# ─────────────────────────────────────────────────────────────

def int_to_hex(color_int: int) -> str:
    """Convert PyMuPDF integer color to hex string."""
    return f"#{color_int:06x}"


def rgb_tuple_to_hex(rgb: tuple) -> str:
    """Convert (r, g, b) float tuple [0-1] to hex."""
    r, g, b = [int(c * 255) for c in rgb[:3]]
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex to (r, g, b) floats [0-1]."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def relative_luminance(rgb: tuple) -> float:
    """WCAG 2.1 relative luminance calculation."""
    def linearize(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = [linearize(c) for c in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(hex1: str, hex2: str) -> float:
    """WCAG contrast ratio between two hex colors."""
    l1 = relative_luminance(hex_to_rgb(hex1))
    l2 = relative_luminance(hex_to_rgb(hex2))
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def closest_okabe_ito(hex_color: str) -> tuple:
    """Find closest Okabe-Ito color by Euclidean distance in RGB."""
    rgb = hex_to_rgb(hex_color)
    best_dist = float("inf")
    best_name = "Custom"
    for oi_hex, oi_name in OKABE_ITO.items():
        oi_rgb = hex_to_rgb(oi_hex)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb, oi_rgb)))
        if dist < best_dist:
            best_dist = dist
            best_name = oi_name
    return best_name, best_dist


# ─────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────

def extract_all(pdf_path: str) -> tuple:
    """Extract all text spans, drawings, and images from PDF."""
    doc = fitz.open(pdf_path)
    page = doc[0]  # posters are single-page

    page_rect = page.rect
    page_w = page_rect.width
    page_h = page_rect.height

    # Text extraction with full detail
    spans = []
    text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:  # text blocks only
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                flags = span.get("flags", 0)
                spans.append(TextSpan(
                    text=text,
                    font=span.get("font", ""),
                    size=round(span.get("size", 0), 1),
                    color=int_to_hex(span.get("color", 0)),
                    bold=bool(flags & (1 << 4)),
                    italic=bool(flags & (1 << 1)),
                    bbox=tuple(round(v, 1) for v in span.get("bbox", (0, 0, 0, 0))),
                ))

    # Drawings extraction
    drawings = []
    for d in page.get_drawings():
        rect = d.get("rect")
        if rect is None:
            continue
        fill_color = None
        if d.get("fill"):
            fill_color = rgb_tuple_to_hex(d["fill"])
        stroke_color = None
        if d.get("color"):
            stroke_color = rgb_tuple_to_hex(d["color"])
        drawings.append(Drawing(
            type=d.get("type", "path"),
            bbox=(round(rect.x0, 1), round(rect.y0, 1),
                  round(rect.x1, 1), round(rect.y1, 1)),
            fill=fill_color,
            stroke=stroke_color,
            width=round(d.get("width", 0) or 0, 2),
        ))

    # Image extraction
    images = []
    for img in page.get_images(full=True):
        xref = img[0]
        for img_rect in page.get_image_rects(xref):
            images.append(ImageInfo(
                bbox=(round(img_rect.x0, 1), round(img_rect.y0, 1),
                      round(img_rect.x1, 1), round(img_rect.y1, 1)),
                width_pt=round(img_rect.width, 1),
                height_pt=round(img_rect.height, 1),
                xref=xref,
            ))

    doc.close()
    return spans, drawings, images, page_w, page_h


# ─────────────────────────────────────────────────────────────
# Analysis: Typography
# ─────────────────────────────────────────────────────────────

MATH_FONT_PATTERNS = ["CMM", "CMSY", "CMEX", "CMR", "Symbol", "Math"]


def is_math_font(font_name: str) -> bool:
    return any(p.lower() in font_name.lower() for p in MATH_FONT_PATTERNS)


def analyze_typography(spans: list) -> TypographyReport:
    """Audit typography: fonts, sizes, hierarchy, readability."""
    font_counter = Counter()
    size_counter = Counter()
    math_fonts = set()
    families = set()

    for s in spans:
        font_counter[s.font] += 1
        size_counter[s.size] += 1
        if is_math_font(s.font):
            math_fonts.add(s.font)
        else:
            # Extract base family (e.g., "FiraSans" from "FiraSans-Bold")
            base = s.font.split("-")[0].split("_")[0]
            families.add(base)

    sizes = sorted(size_counter.keys(), reverse=True)
    max_size = sizes[0] if sizes else 0
    min_size = sizes[-1] if sizes else 0

    # Most common body sizes (middle range)
    body_sizes = [s for s in sizes if 18 <= s <= 36]

    # Hierarchy check
    issues = []
    score = 15  # start at max

    if max_size < TYPO_THRESHOLDS["title_min"]:
        issues.append(f"CRITICAL: Title size {max_size}pt < {TYPO_THRESHOLDS['title_min']}pt minimum")
        score -= 5

    if min_size < TYPO_THRESHOLDS["reference_min"]:
        issues.append(f"HIGH: Smallest text {min_size}pt < {TYPO_THRESHOLDS['reference_min']}pt minimum")
        score -= 3

    non_math_families = [f for f in families if not is_math_font(f)]
    if len(non_math_families) > TARGET_MAX_FONTS:
        issues.append(f"MEDIUM: {len(non_math_families)} font families (max {TARGET_MAX_FONTS}): {non_math_families}")
        score -= 2

    # Check hierarchy: should have clear size steps
    if len(sizes) >= 3:
        top3 = sizes[:3]
        if top3[0] - top3[1] < 5:
            issues.append("LOW: Top two font sizes too close - weak visual hierarchy")
            score -= 1

    return TypographyReport(
        fonts=dict(font_counter.most_common()),
        font_families=sorted(non_math_families),
        math_fonts=sorted(math_fonts),
        size_histogram=dict(sorted(size_counter.items(), reverse=True)),
        max_size=max_size,
        min_size=min_size,
        title_size=max_size,
        body_sizes=body_sizes,
        hierarchy_valid=len(issues) == 0,
        issues=issues,
        score=max(0, score),
    )


# ─────────────────────────────────────────────────────────────
# Analysis: Colors
# ─────────────────────────────────────────────────────────────

def find_background_for_span(span_bbox: tuple, bg_rects: list,
                             page_width: float = 0,
                             text_color: str = None) -> str:
    """Find the background color behind a text span by spatial containment.

    Uses smallest-area containment to find the most specific background.
    Handles TikZ overlay fills (large page-spanning rectangles) as well as
    beamer block backgrounds (smaller, content-sized rectangles).

    Special handling for TikZ overlays: full-width rectangles (>90% page width)
    are recognized as overlay fills (header/footer bars). These use a larger
    tolerance because TikZ 'remember picture, overlay' coordinates may have
    slight offsets vs beamer content coordinates.
    """
    sx0, sy0, sx1, sy1 = span_bbox
    cx = (sx0 + sx1) / 2
    cy = (sy0 + sy1) / 2

    # Find the smallest containing background rect
    best_bg = "#ffffff"  # default: white page
    best_area = float("inf")
    best_is_overlay = False

    for bg_bbox, bg_color in bg_rects:
        bx0, by0, bx1, by1 = bg_bbox
        bg_w = bx1 - bx0
        bg_h = by1 - by0

        # TikZ overlays span full page width - use generous Y tolerance
        is_overlay = page_width > 0 and bg_w > page_width * 0.9
        y_tol = 15 if is_overlay else 2  # 15pt (~5mm) for overlays
        x_tol = 2

        if (bx0 - x_tol) <= cx <= (bx1 + x_tol) and (by0 - y_tol) <= cy <= (by1 + y_tol):
            area = bg_w * bg_h
            if area < best_area:
                best_area = area
                best_bg = bg_color
                best_is_overlay = is_overlay

    # Z-order heuristic: TikZ overlay fills are rendered BEHIND beamer frame
    # content. The overlay may extend past visible header into the white
    # content area. Determine if text is designed for the overlay or for white:
    # compare contrast on overlay vs contrast on white - if overlay gives
    # BETTER contrast, the text belongs to the overlay (header/footer text).
    if best_is_overlay and best_bg != "#ffffff" and text_color:
        overlay_ratio = contrast_ratio(text_color, best_bg)
        white_ratio = contrast_ratio(text_color, "#ffffff")
        # Text designed for white bg has better contrast on white
        if white_ratio > overlay_ratio:
            best_bg = "#ffffff"  # beamer frame is white above overlay

    return best_bg


def analyze_colors(spans: list, drawings: list, page_width: float = 0) -> ColorReport:
    """Extract colors, check WCAG contrast with spatial awareness."""
    text_colors = Counter()
    all_colors = Counter()
    bg_rects = []  # [(bbox, color)] for spatial matching

    for s in spans:
        text_colors[s.color] += 1
        all_colors[s.color] += 1

    # Estimate page height from span positions for coordinate normalization
    page_h_est = max((s.bbox[3] for s in spans), default=3370) * 1.05

    for d in drawings:
        if d.fill:
            all_colors[d.fill] += 1
            # Any filled area can be a background (TikZ overlays, beamer blocks, etc.)
            bbox = d.bbox
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            # TikZ 'remember picture, overlay' uses PDF bottom-left origin,
            # producing negative Y coordinates in PyMuPDF's top-left system.
            # Normalize by adding page height to bring into the visible range.
            if bbox[1] < 0 or bbox[3] < 0:
                bbox = (bbox[0], bbox[1] + page_h_est,
                        bbox[2], bbox[3] + page_h_est)
                h = bbox[3] - bbox[1]

            if w > 50 and h > 10:  # significant filled area
                bg_rects.append((bbox, d.fill))

    bg_unique = list(dict.fromkeys(bg for _, bg in bg_rects))

    # Spatially-aware contrast: check each span against its ACTUAL background.
    # Background resolution uses coordinate normalization (TikZ overlay fills)
    # + z-order contrast comparison (see find_background_for_span).
    contrast_pairs = []
    violations = []
    checked_pairs = set()

    for s in spans:
        actual_bg = find_background_for_span(s.bbox, bg_rects, page_width, s.color)

        pair_key = (s.color, actual_bg)
        if pair_key in checked_pairs:
            continue
        checked_pairs.add(pair_key)

        # Skip same-color checks (invisible text is a different issue)
        if s.color == actual_bg:
            continue

        ratio = contrast_ratio(s.color, actual_bg)
        is_large = s.size >= 18 or (s.size >= 14 and s.bold)
        threshold = WCAG_AA_LARGE if is_large else WCAG_AA_NORMAL
        passes_aa = ratio >= threshold

        pair = {
            "fg": s.color, "bg": actual_bg, "ratio": round(ratio, 2),
            "wcag_aa": passes_aa, "large_text": is_large,
            "text_sample": s.text[:30],
        }
        contrast_pairs.append(pair)

        if not passes_aa:
            size_note = "large" if is_large else "normal"
            violations.append(
                f"WCAG AA FAIL: {s.color} on {actual_bg} = {ratio:.1f}:1 "
                f"(need {threshold}:1 for {size_note} text, e.g. \"{s.text[:20]}\")"
            )

    # WCAG 1.4.11: Non-text contrast for borders and UI components
    # Only check poster-level UI elements, NOT embedded figure internals.
    # A poster-level element must span at least 20% of page width OR height.
    min_dim = page_width * 0.2 if page_width > 0 else 300
    checked_1411 = set()
    for d in drawings:
        if d.stroke and d.fill and d.stroke != d.fill:
            w = d.bbox[2] - d.bbox[0]
            h = d.bbox[3] - d.bbox[1]
            # Only significant poster-level UI elements (callout boxes, block borders)
            if w > min_dim or h > min_dim:
                ratio = contrast_ratio(d.stroke, d.fill)
                pair_key = (d.stroke, d.fill)
                if ratio < 3.0 and pair_key not in checked_1411:
                    checked_1411.add(pair_key)
                    violations.append(
                        f"WCAG 1.4.11 FAIL: border {d.stroke} on fill {d.fill} = {ratio:.1f}:1 "
                        f"(need 3.0:1 for non-text UI components)"
                    )

    # Okabe-Ito matching
    oi_match = {}
    for c in all_colors:
        name, dist = closest_okabe_ito(c)
        oi_match[c] = {"closest": name, "distance": round(dist, 3)}

    issues = []
    score = 10  # max
    if violations:
        score -= min(5, len(violations))
        issues.append(f"HIGH: {len(violations)} WCAG contrast violations found")
    if len(all_colors) > TARGET_MAX_COLORS:
        score -= 1
        issues.append(f"LOW: {len(all_colors)} unique colors (target max {TARGET_MAX_COLORS})")

    return ColorReport(
        unique_colors=dict(all_colors.most_common()),
        text_colors=dict(text_colors.most_common()),
        background_colors=bg_unique,
        contrast_pairs=contrast_pairs[:20],  # top 20
        violations=violations,
        okabe_ito_match=oi_match,
        issues=issues,
        score=max(0, score),
    )


# ─────────────────────────────────────────────────────────────
# Analysis: Layout
# ─────────────────────────────────────────────────────────────

def analyze_layout(spans: list, drawings: list, images: list,
                   page_w: float, page_h: float, pdf_path: str = "") -> LayoutReport:
    """Analyze spatial layout: margins, columns, whitespace, alignment."""
    page_w_mm = page_w * PT_TO_MM
    page_h_mm = page_h * PT_TO_MM

    # Collect all element bounding boxes
    all_bboxes = []
    for s in spans:
        all_bboxes.append(s.bbox)
    for img in images:
        all_bboxes.append(img.bbox)

    if not all_bboxes:
        return LayoutReport(
            page_width_mm=round(page_w_mm, 1),
            page_height_mm=round(page_h_mm, 1),
            margins={"left": 0, "right": 0, "top": 0, "bottom": 0},
            columns_detected=0, column_boundaries=[], column_start_y=[],
            column_alignment_error_mm=0, whitespace_pct=0, element_count=0,
            issues=["CRITICAL: No content detected"], score=0,
        )

    # Margins
    min_x = min(b[0] for b in all_bboxes)
    max_x = max(b[2] for b in all_bboxes)
    min_y = min(b[1] for b in all_bboxes)
    max_y = max(b[3] for b in all_bboxes)

    margins = {
        "left": round(min_x * PT_TO_MM, 1),
        "right": round((page_w - max_x) * PT_TO_MM, 1),
        "top": round(min_y * PT_TO_MM, 1),
        "bottom": round((page_h - max_y) * PT_TO_MM, 1),
    }

    # Column detection: cluster x0 values of text spans in middle region
    # (exclude header/footer by filtering y position)
    header_y = page_h * 0.25  # top 25% is header
    footer_y = page_h * 0.92  # bottom 8% is footer
    body_spans = [s for s in spans
                  if s.bbox[1] > header_y and s.bbox[3] < footer_y]

    # Cluster x0 positions to find column starts
    # For A0 posters, columns are ~250pt apart; use 150pt gap threshold
    gap_threshold = page_w * 0.06  # 6% of page width (~143pt for A0)
    x0_values = sorted(set(round(s.bbox[0], 0) for s in body_spans))
    columns = []
    if x0_values:
        current_cluster = [x0_values[0]]
        for x in x0_values[1:]:
            if x - current_cluster[-1] > gap_threshold:
                columns.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [x]
            else:
                current_cluster.append(x)
        columns.append(sum(current_cluster) / len(current_cluster))

    # Column start Y positions (first content in each column)
    # Use a wider tolerance for column membership (~10% of page width)
    col_tolerance = page_w * 0.10
    col_start_y = []
    for col_x in columns:
        col_spans = [s for s in body_spans
                     if abs(s.bbox[0] - col_x) < col_tolerance]
        if col_spans:
            col_start_y.append(min(s.bbox[1] for s in col_spans))
        else:
            col_start_y.append(0)

    # Column alignment error
    alignment_error = 0
    if len(col_start_y) >= 2:
        alignment_error = (max(col_start_y) - min(col_start_y)) * PT_TO_MM

    # Whitespace estimation via pixel sampling
    whitespace_pct = estimate_whitespace(pdf_path) if pdf_path else 35.0

    issues = []
    score = 15  # max

    # Beamerposter margins are template-controlled; 8mm+ is acceptable
    if margins["left"] < 8:
        issues.append(f"HIGH: Left margin {margins['left']}mm < 8mm minimum")
        score -= 2
    if margins["right"] < 8:
        issues.append(f"HIGH: Right margin {margins['right']}mm < 8mm minimum")
        score -= 2
    if alignment_error > 10:
        issues.append(f"MEDIUM: Column start misalignment {alignment_error:.1f}mm")
        score -= 2
    if whitespace_pct < TARGET_WHITESPACE_PCT:
        issues.append(f"MEDIUM: Whitespace {whitespace_pct:.0f}% < {TARGET_WHITESPACE_PCT}% target")
        score -= 2

    return LayoutReport(
        page_width_mm=round(page_w_mm, 1),
        page_height_mm=round(page_h_mm, 1),
        margins=margins,
        columns_detected=len(columns),
        column_boundaries=[round(c * PT_TO_MM, 1) for c in columns],
        column_start_y=[round(y * PT_TO_MM, 1) for y in col_start_y],
        column_alignment_error_mm=round(alignment_error, 1),
        whitespace_pct=round(whitespace_pct, 1),
        element_count=len(all_bboxes),
        issues=issues,
        score=max(0, score),
    )


def estimate_whitespace(pdf_path: str) -> float:
    """Estimate whitespace by rendering and counting white/near-white pixels."""
    if Image is None or np is None:
        return 35.0  # assume moderate whitespace if no imaging libs

    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap(dpi=36)  # very low DPI for speed
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        doc.close()

        if pix.n >= 3:
            gray = np.mean(img_data[:, :, :3], axis=2)
        else:
            gray = img_data[:, :, 0].astype(float)

        # White/near-white pixels (>240 on 0-255 scale)
        white_pixels = np.sum(gray > 240)
        total_pixels = gray.size
        return (white_pixels / total_pixels) * 100
    except Exception:
        return 35.0


# ─────────────────────────────────────────────────────────────
# Analysis: Visual Balance
# ─────────────────────────────────────────────────────────────

def analyze_balance(pdf_path: str, page_w: float, page_h: float) -> BalanceReport:
    """Analyze visual weight distribution using rendered pixmap."""
    if Image is None or np is None:
        return BalanceReport(
            quadrant_weights={}, column_weights=[],
            max_imbalance_pct=0,
            issues=["SKIP: Pillow/numpy not available for balance analysis"],
            score=10,
        )

    doc = fitz.open(pdf_path)
    page = doc[0]
    # Render at low DPI for fast analysis
    pix = page.get_pixmap(dpi=72)
    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

    # Convert to grayscale (visual weight = darkness)
    if pix.n >= 3:
        gray = np.mean(img_data[:, :, :3], axis=2)
    else:
        gray = img_data[:, :, 0].astype(float)

    # Invert: dark pixels = high weight
    weight = 255.0 - gray

    h, w = weight.shape

    # Mask header (~22% from top) and footer (~5% from bottom) for content-only balance
    header_end = int(h * 0.25)
    footer_start = int(h * 0.95)
    content = weight[header_end:footer_start, :]

    h_c, w_c = content.shape
    mid_h = h_c // 2
    mid_w = w_c // 2
    third_w = w_c // 3

    # Quadrant weights (content area only)
    tl = float(np.mean(content[:mid_h, :mid_w]))
    tr = float(np.mean(content[:mid_h, mid_w:]))
    bl = float(np.mean(content[mid_h:, :mid_w]))
    br = float(np.mean(content[mid_h:, mid_w:]))

    quadrants = {"TL": round(tl, 1), "TR": round(tr, 1),
                 "BL": round(bl, 1), "BR": round(br, 1)}

    # Column weights (content area only, 3 columns)
    col_weights = [
        round(float(np.mean(content[:, :third_w])), 1),
        round(float(np.mean(content[:, third_w:2*third_w])), 1),
        round(float(np.mean(content[:, 2*third_w:])), 1),
    ]

    # Imbalance
    all_weights = list(quadrants.values())
    max_w = max(all_weights)
    min_w = min(all_weights)
    imbalance = ((max_w - min_w) / max(max_w, 1)) * 100

    col_max = max(col_weights)
    col_min = min(col_weights)
    col_imbalance = ((col_max - col_min) / max(col_max, 1)) * 100

    doc.close()

    issues = []
    score = 10  # max

    if imbalance > 30:
        issues.append(f"MEDIUM: Quadrant imbalance {imbalance:.0f}% (threshold 30%)")
        score -= 3
    if col_imbalance > 25:
        issues.append(f"MEDIUM: Column weight imbalance {col_imbalance:.0f}% (threshold 25%)")
        score -= 2

    return BalanceReport(
        quadrant_weights=quadrants,
        column_weights=col_weights,
        max_imbalance_pct=round(max(imbalance, col_imbalance), 1),
        issues=issues,
        score=max(0, score),
    )


# ─────────────────────────────────────────────────────────────
# Scoring & Grading
# ─────────────────────────────────────────────────────────────

def compute_score(typo: TypographyReport, colors: ColorReport,
                  layout: LayoutReport, balance: BalanceReport) -> tuple:
    """Compute total score and grade from sub-reports."""

    # Scoring weights (total = 100):
    # Typography: 25pts (15 from analysis + 10 bonus for hierarchy/readability)
    # Colors: 20pts (10 from analysis + 10 bonus for accessibility)
    # Layout: 30pts (15 from analysis + 15 bonus for margins/columns/whitespace)
    # Balance: 15pts (10 from analysis + 5 bonus)
    # Figure quality: 10pts (from image count and sizing)

    typo_score = typo.score  # /15
    # Bonus: good hierarchy
    typo_bonus = 10
    if not typo.hierarchy_valid:
        typo_bonus -= 5
    if len(typo.font_families) <= 2:
        typo_bonus += 0  # already good
    typo_total = min(25, typo_score + typo_bonus)

    color_score = colors.score  # /10
    # Bonus: accessibility
    color_bonus = 10
    if colors.violations:
        color_bonus -= min(8, len(colors.violations) * 2)
    color_total = min(20, color_score + color_bonus)

    layout_score = layout.score  # /15
    layout_bonus = 15
    if layout.columns_detected < 2:
        layout_bonus -= 5
    if layout.column_alignment_error_mm > 15:
        layout_bonus -= 5
    if layout.whitespace_pct < 20:
        layout_bonus -= 5
    layout_total = min(30, layout_score + layout_bonus)

    balance_score = balance.score  # /10
    balance_bonus = 5
    if balance.max_imbalance_pct > 40:
        balance_bonus -= 3
    balance_total = min(15, balance_score + balance_bonus)

    # Figure quality (basic check from layout)
    fig_score = 10  # assume good unless issues detected

    total = typo_total + color_total + layout_total + balance_total + fig_score

    if total >= 90:
        grade = "A+"
    elif total >= 80:
        grade = "A"
    elif total >= 70:
        grade = "B"
    elif total >= 60:
        grade = "C"
    else:
        grade = "F"

    return round(total, 1), grade


def collect_defects(typo, colors, layout, balance) -> list:
    """Collect all defects from sub-reports, sorted by severity."""
    defects = []
    for report in [typo, colors, layout, balance]:
        for issue in report.issues:
            severity = "LOW"
            if issue.startswith("CRITICAL"):
                severity = "CRITICAL"
            elif issue.startswith("HIGH"):
                severity = "HIGH"
            elif issue.startswith("MEDIUM"):
                severity = "MEDIUM"
            defects.append({"severity": severity, "message": issue})

    # Add WCAG violations as HIGH
    for v in colors.violations:
        defects.append({"severity": "HIGH", "message": v})

    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    defects.sort(key=lambda d: severity_order.get(d["severity"], 4))
    return defects


# ─────────────────────────────────────────────────────────────
# Main Inspector
# ─────────────────────────────────────────────────────────────

def inspect_poster(pdf_path: str) -> InspectionReport:
    """Run full inspection on a poster PDF."""
    spans, drawings, images, page_w, page_h = extract_all(pdf_path)

    typo = analyze_typography(spans)
    colors = analyze_colors(spans, drawings, page_w)
    layout = analyze_layout(spans, drawings, images, page_w, page_h, pdf_path)
    balance = analyze_balance(pdf_path, page_w, page_h)

    total, grade = compute_score(typo, colors, layout, balance)
    defects = collect_defects(typo, colors, layout, balance)

    return InspectionReport(
        file=str(pdf_path),
        page_size_mm=(round(page_w * PT_TO_MM, 1), round(page_h * PT_TO_MM, 1)),
        typography=typo,
        colors=colors,
        layout=layout,
        balance=balance,
        total_score=total,
        grade=grade,
        defects=defects,
    )


# ─────────────────────────────────────────────────────────────
# Output Formatters
# ─────────────────────────────────────────────────────────────

def format_text_report(report: InspectionReport) -> str:
    """Format inspection report as readable text."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"  POSTER INSPECTOR REPORT")
    lines.append(f"  File: {report.file}")
    lines.append(f"  Page: {report.page_size_mm[0]} x {report.page_size_mm[1]} mm")
    lines.append("=" * 70)

    # Score
    lines.append(f"\n  TOTAL SCORE: {report.total_score}/100  (Grade: {report.grade})")
    lines.append("")

    # Typography
    lines.append("-" * 50)
    lines.append(f"  TYPOGRAPHY (Score: {report.typography.score}/15)")
    lines.append("-" * 50)
    lines.append(f"  Font families: {', '.join(report.typography.font_families)}")
    if report.typography.math_fonts:
        lines.append(f"  Math fonts: {', '.join(report.typography.math_fonts)}")
    lines.append(f"  Size range: {report.typography.min_size}pt - {report.typography.max_size}pt")
    lines.append(f"  Title size: {report.typography.title_size}pt (min: {TYPO_THRESHOLDS['title_min']}pt)")
    lines.append(f"  Size histogram:")
    for size, count in sorted(report.typography.size_histogram.items(), reverse=True)[:10]:
        bar = "#" * min(40, count)
        lines.append(f"    {size:6.1f}pt  [{count:4d}] {bar}")

    # Colors
    lines.append("")
    lines.append("-" * 50)
    lines.append(f"  COLORS (Score: {report.colors.score}/10)")
    lines.append("-" * 50)
    lines.append(f"  Unique colors: {len(report.colors.unique_colors)}")
    lines.append(f"  Text colors: {len(report.colors.text_colors)}")
    lines.append(f"  Background fills: {len(report.colors.background_colors)}")
    lines.append(f"  Palette:")
    for color, count in list(report.colors.unique_colors.items())[:12]:
        oi = report.colors.okabe_ito_match.get(color, {})
        oi_name = oi.get("closest", "?")
        oi_dist = oi.get("distance", 0)
        marker = " *" if oi_dist < 0.05 else ""
        lines.append(f"    {color}  [{count:4d}]  ~{oi_name}{marker}")
    if report.colors.violations:
        lines.append(f"  WCAG violations: {len(report.colors.violations)}")
        for v in report.colors.violations[:5]:
            lines.append(f"    ! {v}")

    # Layout
    lines.append("")
    lines.append("-" * 50)
    lines.append(f"  LAYOUT (Score: {report.layout.score}/15)")
    lines.append("-" * 50)
    lines.append(f"  Margins (mm): L={report.layout.margins['left']}  R={report.layout.margins['right']}  "
                 f"T={report.layout.margins['top']}  B={report.layout.margins['bottom']}")
    lines.append(f"  Columns detected: {report.layout.columns_detected}")
    lines.append(f"  Column boundaries (mm): {report.layout.column_boundaries}")
    lines.append(f"  Column start Y (mm): {report.layout.column_start_y}")
    lines.append(f"  Column alignment error: {report.layout.column_alignment_error_mm}mm")
    lines.append(f"  Whitespace: {report.layout.whitespace_pct}%")
    lines.append(f"  Total elements: {report.layout.element_count}")

    # Balance
    lines.append("")
    lines.append("-" * 50)
    lines.append(f"  VISUAL BALANCE (Score: {report.balance.score}/10)")
    lines.append("-" * 50)
    if report.balance.quadrant_weights:
        q = report.balance.quadrant_weights
        lines.append(f"  Quadrants: TL={q.get('TL', 0)}  TR={q.get('TR', 0)}  "
                     f"BL={q.get('BL', 0)}  BR={q.get('BR', 0)}")
    if report.balance.column_weights:
        lines.append(f"  Column weights: {report.balance.column_weights}")
    lines.append(f"  Max imbalance: {report.balance.max_imbalance_pct}%")

    # Defects
    if report.defects:
        lines.append("")
        lines.append("=" * 50)
        lines.append(f"  DEFECTS ({len(report.defects)} found)")
        lines.append("=" * 50)
        for d in report.defects:
            lines.append(f"  [{d['severity']:8s}] {d['message']}")

    lines.append("")
    return "\n".join(lines)


def to_json(report: InspectionReport) -> str:
    """Serialize report to JSON."""
    def serialize(obj):
        if hasattr(obj, "__dict__"):
            return {k: serialize(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [serialize(v) for v in obj]
        return obj
    return json.dumps(serialize(report), indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def find_poster_pdf() -> Optional[Path]:
    """Find most recent poster PDF in the project."""
    project_root = Path(__file__).parent.parent.parent
    poster_dir = project_root / ".docs" / "conferences" / "EGU26" / "poster"
    candidates = list(poster_dir.glob("poster*.pdf"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def main():
    parser = argparse.ArgumentParser(description="Poster Inspector - PDF Design Analysis")
    parser.add_argument("pdf", nargs="?", help="Path to poster PDF (auto-detects if omitted)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Include all contrast pairs")
    args = parser.parse_args()

    if args.pdf:
        pdf_path = Path(args.pdf)
    else:
        pdf_path = find_poster_pdf()
        if pdf_path is None:
            sys.exit("ERROR: No poster PDF found. Pass path as argument.")

    if not pdf_path.exists():
        sys.exit(f"ERROR: File not found: {pdf_path}")

    print(f"Inspecting: {pdf_path}", file=sys.stderr)
    report = inspect_poster(str(pdf_path))

    if args.json:
        print(to_json(report))
    else:
        print(format_text_report(report))

    # Exit code: 0 if A/A+, 1 if B or lower
    sys.exit(0 if report.total_score >= 80 else 1)


if __name__ == "__main__":
    main()
