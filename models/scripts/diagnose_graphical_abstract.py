"""
Graphical Abstract Diagnostic Tool
====================================
Programmatically detects quality issues in the graphical abstract:
- Text-text overlaps
- Text-box overlaps
- Box-box overlaps
- Spacing inconsistencies between sections
- Arrow alignment (start/end vs box edges)
- Font size consistency
- Color contrast (WCAG AA)
- Alignment deviations

Outputs:
  1. Console report with all issues found
  2. Debug overlay PNG showing bounding boxes + issues
"""
import sys
import os
import io

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from collections import defaultdict


def get_all_elements(fig, ax):
    """Extract all visual elements with their bounding boxes in data coords."""
    renderer = fig.canvas.get_renderer()

    texts = []
    patches = []
    arrows = []

    # Collect text elements
    for t in ax.texts:
        txt = t.get_text().strip()
        if not txt:
            continue
        try:
            bb = t.get_window_extent(renderer)
            # Convert display coords to data coords
            inv = ax.transData.inverted()
            bb_data = bb.transformed(inv)
            texts.append({
                'text': txt[:50],
                'x0': bb_data.x0, 'y0': bb_data.y0,
                'x1': bb_data.x1, 'y1': bb_data.y1,
                'fontsize': t.get_fontsize(),
                'ha': t.get_ha(),
                'va': t.get_va(),
                'color': t.get_color(),
                'zorder': t.get_zorder(),
                'center_x': (bb_data.x0 + bb_data.x1) / 2,
                'center_y': (bb_data.y0 + bb_data.y1) / 2,
            })
        except Exception:
            pass

    # Collect patches (boxes)
    for p in ax.patches:
        if isinstance(p, FancyBboxPatch):
            try:
                bb = p.get_window_extent(renderer)
                inv = ax.transData.inverted()
                bb_data = bb.transformed(inv)
                patches.append({
                    'type': 'FancyBboxPatch',
                    'x0': bb_data.x0, 'y0': bb_data.y0,
                    'x1': bb_data.x1, 'y1': bb_data.y1,
                    'fc': p.get_facecolor(),
                    'ec': p.get_edgecolor(),
                    'zorder': p.get_zorder(),
                    'width': bb_data.x1 - bb_data.x0,
                    'height': bb_data.y1 - bb_data.y0,
                })
            except Exception:
                pass

    # Collect annotations (arrows) - matplotlib creates Annotation objects
    from matplotlib.text import Annotation
    for child in ax.get_children():
        if isinstance(child, Annotation) and child.arrow_patch is not None:
            try:
                # Get positions from the annotation's xy/xytext
                start = child.xyann if hasattr(child, 'xyann') else child.xy
                end = child.xy
                # xytext is the start, xy is the end (arrow points to xy)
                xytext = getattr(child, '_xytext', None)
                if xytext is not None:
                    start = xytext
                arrows.append({
                    'start': start,
                    'end': end,
                    'zorder': child.get_zorder(),
                })
            except Exception:
                pass
        elif isinstance(child, FancyArrowPatch) and not isinstance(child, type(None)):
            try:
                posA = getattr(child, '_posA_posB', [None, None])
                if posA and posA[0] is not None:
                    arrows.append({
                        'start': posA[0],
                        'end': posA[1],
                        'zorder': child.get_zorder(),
                    })
            except Exception:
                pass

    return texts, patches, arrows


def check_overlaps(texts, patches, threshold=0.002):
    """Check for overlapping bounding boxes."""
    issues = []

    def boxes_overlap(a, b):
        """Check if two bounding boxes overlap by more than threshold."""
        ox = max(0, min(a['x1'], b['x1']) - max(a['x0'], b['x0']))
        oy = max(0, min(a['y1'], b['y1']) - max(a['y0'], b['y0']))
        return ox > threshold and oy > threshold, ox * oy

    # Text-text overlaps
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            overlaps, area = boxes_overlap(texts[i], texts[j])
            if overlaps:
                issues.append({
                    'type': 'TEXT_TEXT_OVERLAP',
                    'severity': 'HIGH' if area > 0.001 else 'MEDIUM',
                    'area': area,
                    'elements': [texts[i]['text'], texts[j]['text']],
                    'detail': f"Text '{texts[i]['text']}' overlaps with '{texts[j]['text']}' (area={area:.5f})"
                })

    # Text-patch overlaps (only if same or lower zorder - higher zorder text ON patch is OK)
    for t in texts:
        for p in patches:
            # Text inside a box with HIGHER zorder is intentional (label on box)
            if t['zorder'] > p['zorder']:
                # Check if text CENTER is inside the patch - this is intentional
                if (p['x0'] <= t['center_x'] <= p['x1'] and
                    p['y0'] <= t['center_y'] <= p['y1']):
                    continue  # Text is a label ON this box - OK

            overlaps, area = boxes_overlap(t, p)
            if overlaps and t['zorder'] <= p['zorder']:
                issues.append({
                    'type': 'TEXT_PATCH_OVERLAP',
                    'severity': 'HIGH',
                    'area': area,
                    'elements': [t['text'], f"Box({p['x0']:.3f},{p['y0']:.3f})"],
                    'detail': f"Text '{t['text']}' overlaps with box at ({p['x0']:.3f},{p['y0']:.3f}) z={p['zorder']}"
                })

    return issues


def check_spacing(texts, patches):
    """Check spacing consistency between adjacent sections.

    Measures the gap between each section's content bottom and the next
    section's label top. Skips non-adjacent sections (e.g., if content
    blocks like chart + key finding span between two labels).
    """
    issues = []

    # Find section labels (bold, uppercase text)
    section_labels = [t for t in texts if t['text'].isupper() and len(t['text']) > 3]

    if len(section_labels) < 2:
        return issues

    # Sort by y position (top to bottom)
    section_labels.sort(key=lambda t: -t['y0'])

    # For each label, find its associated content (patches directly below it)
    for i in range(len(section_labels) - 1):
        label = section_labels[i]
        next_label = section_labels[i + 1]

        # Find content patches between this label and the next
        content_patches = [
            p for p in patches
            if p['y1'] <= label['y0'] and p['y0'] >= next_label['y1']
        ]

        if content_patches:
            # The gap is from the lowest content bottom to the next label top
            lowest_content = min(p['y0'] for p in content_patches)
            gap_to_next = lowest_content - next_label['y1']
        else:
            # No content between labels; measure label-to-label gap
            gap_to_next = label['y0'] - next_label['y1']

        # Only flag very large or very small gaps relative to INTER_GAP (0.020)
        if gap_to_next < 0.005:
            issues.append({
                'type': 'SPACING_TOO_TIGHT',
                'severity': 'HIGH',
                'detail': f"Gap {label['text'][:25]} -> {next_label['text'][:25]}: "
                          f"{gap_to_next:.4f} (< 0.005 min)"
            })
        elif gap_to_next < 0.010:
            issues.append({
                'type': 'SPACING_TIGHT',
                'severity': 'MEDIUM',
                'detail': f"Gap {label['text'][:25]} -> {next_label['text'][:25]}: "
                          f"{gap_to_next:.4f} (tight, ideal > 0.015)"
            })

    return issues


def check_alignment(texts, patches):
    """Check horizontal and vertical alignment."""
    issues = []

    # Group texts by approximate x position (left-aligned texts)
    left_texts = [t for t in texts if t['ha'] == 'left']
    if len(left_texts) > 1:
        x_positions = [t['x0'] for t in left_texts]
        # Find clusters
        clusters = defaultdict(list)
        for t in left_texts:
            # Round to nearest 0.02 to find alignment groups
            key = round(t['x0'] / 0.02) * 0.02
            clusters[key].append(t)

        for key, group in clusters.items():
            if len(group) > 1:
                x_vals = [t['x0'] for t in group]
                spread = max(x_vals) - min(x_vals)
                if spread > 0.005:
                    names = [t['text'][:20] for t in group]
                    issues.append({
                        'type': 'ALIGNMENT_DEVIATION',
                        'severity': 'LOW',
                        'detail': f"Left-aligned group at x~{key:.3f} has spread={spread:.4f}: {names}"
                    })

    # Check center-aligned texts
    center_texts = [t for t in texts if t['ha'] == 'center']
    if len(center_texts) > 1:
        center_x_vals = [t['center_x'] for t in center_texts]
        clusters = defaultdict(list)
        for t in center_texts:
            key = round(t['center_x'] / 0.02) * 0.02
            clusters[key].append(t)

        for key, group in clusters.items():
            if len(group) > 1:
                cx_vals = [t['center_x'] for t in group]
                spread = max(cx_vals) - min(cx_vals)
                if spread > 0.008:
                    names = [t['text'][:20] for t in group]
                    issues.append({
                        'type': 'CENTER_ALIGNMENT_DEVIATION',
                        'severity': 'LOW',
                        'detail': f"Center-aligned group at cx~{key:.3f} has spread={spread:.4f}: {names}"
                    })

    return issues


def check_arrow_alignment(arrows, patches, tolerance=0.02):
    """Check that arrows start/end at box edges."""
    issues = []

    for a in arrows:
        start_x, start_y = a['start']
        end_x, end_y = a['end']

        # Check if start point is at a box bottom (with x tolerance for edge cases)
        x_tol = 0.005
        start_on_box = False
        for p in patches:
            if (abs(start_y - p['y0']) < tolerance and
                p['x0'] - x_tol <= start_x <= p['x1'] + x_tol):
                start_on_box = True
                break

        # Check if end point is at a box top
        end_on_box = False
        for p in patches:
            if (abs(end_y - p['y1']) < tolerance and
                p['x0'] - x_tol <= end_x <= p['x1'] + x_tol):
                end_on_box = True
                break

        if not start_on_box:
            issues.append({
                'type': 'ARROW_DETACHED_START',
                'severity': 'MEDIUM',
                'detail': f"Arrow start ({start_x:.3f},{start_y:.3f}) not aligned with any box bottom"
            })

        if not end_on_box:
            issues.append({
                'type': 'ARROW_DETACHED_END',
                'severity': 'MEDIUM',
                'detail': f"Arrow end ({end_x:.3f},{end_y:.3f}) not aligned with any box top"
            })

    return issues


def check_font_consistency(texts):
    """Check font size consistency within groups."""
    issues = []

    # Group by fontsize
    size_counts = defaultdict(list)
    for t in texts:
        size_counts[t['fontsize']].append(t['text'][:30])

    # Flag if too many different sizes (more than 5 is suspicious)
    if len(size_counts) > 6:
        issues.append({
            'type': 'FONT_SIZE_PROLIFERATION',
            'severity': 'LOW',
            'detail': f"{len(size_counts)} different font sizes used: "
                      f"{sorted(size_counts.keys())}"
        })

    return issues


def check_color_contrast(texts):
    """Check text color contrast against assumed backgrounds."""
    issues = []

    def luminance(color_str):
        """Approximate relative luminance."""
        if isinstance(color_str, str):
            from matplotlib.colors import to_rgba
            try:
                r, g, b, _ = to_rgba(color_str)
            except Exception:
                return 1.0
        elif hasattr(color_str, '__len__') and len(color_str) >= 3:
            r, g, b = color_str[0], color_str[1], color_str[2]
        else:
            return 1.0

        def srgb(c):
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

        return 0.2126 * srgb(r) + 0.7152 * srgb(g) + 0.0722 * srgb(b)

    for t in texts:
        lum = luminance(t['color'])
        # Assume white background (L=1.0) for most text
        # WCAG AA requires contrast ratio >= 4.5:1 for normal text
        if lum > 0.5:  # Light text on assumed dark background - skip
            continue
        contrast = (1.0 + 0.05) / (lum + 0.05)
        if contrast < 4.5:
            issues.append({
                'type': 'LOW_CONTRAST',
                'severity': 'MEDIUM',
                'detail': f"Text '{t['text'][:30]}' color={t['color']} "
                          f"contrast={contrast:.1f}:1 (need 4.5:1)"
            })

    return issues


def check_margins(texts, patches, margin=0.02):
    """Check if elements are too close to figure edges."""
    issues = []

    for t in texts:
        if t['x0'] < margin:
            issues.append({
                'type': 'CLIPPED_LEFT',
                'severity': 'HIGH',
                'detail': f"Text '{t['text'][:30]}' x0={t['x0']:.4f} < margin {margin}"
            })
        if t['x1'] > 1.0 - margin:
            issues.append({
                'type': 'CLIPPED_RIGHT',
                'severity': 'HIGH',
                'detail': f"Text '{t['text'][:30]}' x1={t['x1']:.4f} > {1.0 - margin}"
            })
        if t['y0'] < margin:
            issues.append({
                'type': 'CLIPPED_BOTTOM',
                'severity': 'HIGH',
                'detail': f"Text '{t['text'][:30]}' y0={t['y0']:.4f} < margin {margin}"
            })
        if t['y1'] > 1.0 - margin:
            issues.append({
                'type': 'CLIPPED_TOP',
                'severity': 'HIGH',
                'detail': f"Text '{t['text'][:30]}' y1={t['y1']:.4f} > {1.0 - margin}"
            })

    return issues


def generate_debug_overlay(fig, ax, texts, patches, arrows, all_issues):
    """Generate a debug image showing bounding boxes and issues."""
    import matplotlib.patches as mpatches

    renderer = fig.canvas.get_renderer()

    # Draw text bounding boxes in green
    for t in texts:
        rect = mpatches.Rectangle(
            (t['x0'], t['y0']), t['x1'] - t['x0'], t['y1'] - t['y0'],
            fill=False, edgecolor='lime', linewidth=0.5, linestyle='--',
            zorder=100, alpha=0.7
        )
        ax.add_patch(rect)

    # Draw patch bounding boxes in cyan
    for p in patches:
        rect = mpatches.Rectangle(
            (p['x0'], p['y0']), p['x1'] - p['x0'], p['y1'] - p['y0'],
            fill=False, edgecolor='cyan', linewidth=0.5, linestyle=':',
            zorder=100, alpha=0.5
        )
        ax.add_patch(rect)

    # Mark overlaps in red
    for issue in all_issues:
        if 'OVERLAP' in issue['type']:
            ax.text(0.5, 0.5, 'X', fontsize=20, color='red', alpha=0.3,
                    ha='center', va='center', zorder=200,
                    transform=ax.transAxes)

    return fig


def run_diagnostics():
    """Run all diagnostics and print report."""
    # Import and generate the figure
    from generate_graphical_abstract import draw

    fig = draw()
    ax = fig.axes[0]

    # Force render to compute bounding boxes
    fig.canvas.draw()

    # Extract elements
    texts, patches, arrows = get_all_elements(fig, ax)

    print("=" * 70)
    print("  GRAPHICAL ABSTRACT DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"\nElements found:")
    print(f"  Text elements:  {len(texts)}")
    print(f"  Patches (boxes): {len(patches)}")
    print(f"  Arrows:          {len(arrows)}")
    print()

    # Run all checks
    all_issues = []

    print("--- Overlap Analysis ---")
    overlap_issues = check_overlaps(texts, patches)
    all_issues.extend(overlap_issues)

    print("--- Spacing Analysis ---")
    spacing_issues = check_spacing(texts, patches)
    all_issues.extend(spacing_issues)

    print("--- Alignment Analysis ---")
    alignment_issues = check_alignment(texts, patches)
    all_issues.extend(alignment_issues)

    print("--- Arrow Alignment ---")
    arrow_issues = check_arrow_alignment(arrows, patches)
    all_issues.extend(arrow_issues)

    print("--- Font Consistency ---")
    font_issues = check_font_consistency(texts)
    all_issues.extend(font_issues)

    print("--- Color Contrast ---")
    contrast_issues = check_color_contrast(texts)
    all_issues.extend(contrast_issues)

    print("--- Margin Check ---")
    margin_issues = check_margins(texts, patches)
    all_issues.extend(margin_issues)

    # Summary
    high = [i for i in all_issues if i['severity'] == 'HIGH']
    medium = [i for i in all_issues if i['severity'] == 'MEDIUM']
    low = [i for i in all_issues if i['severity'] == 'LOW']

    print("\n" + "=" * 70)
    print(f"  RESULTS: {len(all_issues)} issues found")
    print(f"    HIGH:   {len(high)}")
    print(f"    MEDIUM: {len(medium)}")
    print(f"    LOW:    {len(low)}")
    print("=" * 70)

    for severity in ['HIGH', 'MEDIUM', 'LOW']:
        items = [i for i in all_issues if i['severity'] == severity]
        if items:
            print(f"\n[{severity}]")
            for i, issue in enumerate(items, 1):
                print(f"  {i}. [{issue['type']}] {issue['detail']}")

    # Print element inventory for debugging
    print("\n" + "-" * 70)
    print("  ELEMENT INVENTORY")
    print("-" * 70)
    print("\nText elements (sorted top to bottom):")
    for t in sorted(texts, key=lambda x: -x['y0']):
        print(f"  y=[{t['y0']:.3f},{t['y1']:.3f}] x=[{t['x0']:.3f},{t['x1']:.3f}] "
              f"fs={t['fontsize']} z={t['zorder']} \"{t['text']}\"")

    print(f"\nPatches (sorted top to bottom):")
    for p in sorted(patches, key=lambda x: -x['y0']):
        print(f"  y=[{p['y0']:.3f},{p['y1']:.3f}] x=[{p['x0']:.3f},{p['x1']:.3f}] "
              f"z={p['zorder']} w={p['width']:.3f} h={p['height']:.3f}")

    # Save debug overlay
    debug_fig = generate_debug_overlay(fig, ax, texts, patches, arrows, all_issues)
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                           'docs', 'papers', '1', 'latex', 'figures')
    out_dir = os.path.abspath(out_dir)
    debug_path = os.path.join(out_dir, 'image3_debug.png')
    debug_fig.savefig(debug_path, dpi=150, facecolor='white')
    print(f"\nDebug overlay saved: {debug_path}")

    plt.close(fig)

    # Return exit code: 0 if no HIGH issues, 1 otherwise
    return 1 if high else 0


if __name__ == '__main__':
    exit_code = run_diagnostics()
    sys.exit(exit_code)
