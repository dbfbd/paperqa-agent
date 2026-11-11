"""Conservative fallback for digitally encoded, horizontally ruled PDF tables.

Preserves glyph x/y layout rather than guessing merged header semantics. It does
not OCR, join pages, or claim that every extracted table is semantically correct.
"""

import re
from collections import defaultdict
from statistics import median

import pymupdf


def _layout_text(page, rect, rules):
    words = [w for w in page.get_text("words") if rect.contains(pymupdf.Rect(w[:4]))]
    rows = []
    for word in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        center = (word[1] + word[3]) / 2
        if not rows or abs(rows[-1][0] - center) > 3:
            rows.append((center, [word]))
        else:
            rows[-1][1].append(word)
    # One grid scale for every row keeps grouped headings over numeric columns.
    pitch = max(1.4, median(w[3] - w[1] for w in words) * 0.28) if words else 3
    lines = []
    for y, row in rows:
        segments = []
        for word in sorted(row, key=lambda w: w[0]):
            if segments and word[0] - segments[-1][1] <= (word[3] - word[1]) * 0.65:
                segments[-1][1] = word[2]
                segments[-1][2] += " " + word[4]
            else:
                segments.append([word[0], word[2], word[4]])
        line = ""
        for left, right, value in segments:
            column = round(((left + right) / 2 - rect.x0) / pitch - len(value) / 2)
            line += " " * max(1 if line else 0, column - len(line)) + value
        lines.append((y, line.rstrip()))
    lines.extend((r[1], "-" * round(rect.width / pitch)) for r in rules)
    return "\n".join(text for _, text in sorted(lines)), rows


def recover_ruled_tables(page):
    """Return bounded table regions with source-position text and nearby caption.

    Require >=3 near-identical horizontal rules, >=3 numerical rows and a nearby
    explicit Table caption. No table-specific filenames, values or headers.
    """
    groups = defaultdict(list)
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] != "l":
                continue
            a, b = item[1:3]
            x0, x1 = sorted((a.x, b.x))
            if abs(a.y - b.y) <= 1 and x1 - x0 >= page.rect.width * 0.3:
                key = (round(x0 / 3), round(x1 / 3))
                groups[key].append((x0, (a.y + b.y) / 2, x1))
    captions = [b for b in page.get_text("blocks") if re.match(r"\s*Table\s+\d+\s*[:.]", b[4])]
    result = []
    for rules in groups.values():
        if len({round(r[1], 1) for r in rules}) < 3:
            continue
        x0, x1 = min(r[0] for r in rules), max(r[2] for r in rules)
        matching = [c for c in captions if min(x1, c[2]) - max(x0, c[0]) >= min(x1 - x0, c[2] - c[0]) * 0.7]
        for caption in matching:
            above = [r for r in rules if r[1] < caption[1]]
            below = [r for r in rules if r[1] > caption[3]]
            gap_above = caption[1] - max((r[1] for r in above), default=-1e6)
            gap_below = min((r[1] for r in below), default=1e6) - caption[3]
            if min(gap_above, gap_below) > 30:
                continue
            if gap_above <= gap_below:
                lower = max((c[3] for c in matching if c[3] < caption[1]), default=-1e6)
                bounded = [r for r in above if r[1] > lower]
            else:
                upper = min((c[1] for c in matching if c[1] > caption[3]), default=1e6)
                bounded = [r for r in below if r[1] < upper]
            if len({round(r[1], 1) for r in bounded}) >= 3:
                recovered = _recover_region(page, x0, x1, bounded, caption[4])
                if recovered:
                    result.append(recovered)
    return result


def _recover_region(page, x0, x1, rules, caption):
    top, bottom = min(r[1] for r in rules), max(r[1] for r in rules)
    if not 25 <= bottom - top <= page.rect.height * 0.6:
        return None
    rect = pymupdf.Rect(x0 - 2, top - 2, x1 + 2, bottom + 2)
    text, rows = _layout_text(page, rect, rules)
    numeric_rows = sum(
        sum(bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", w[4])) for w in row) >= 3
        for _, row in rows
    )
    if numeric_rows < 3:
        return None
    # 识别表区不等于恢复了公式；上标可能被摊平为普通数字，遇到时保守标记不确定。
    superscripts = any(
        span.get("flags", 0) & 1
        for block in page.get_text("dict", clip=rect)["blocks"]
        for line in block.get("lines", []) for span in line.get("spans", [])
    )
    return {
        "bbox": list(rect), "method": "ruled-position-text", "numeric_rows": numeric_rows,
        "layout_uncertain": superscripts,
        "text": "[按PDF字符坐标保留的表格，空格体现列位置；多级表头及条件仍需核对]\n"
                + ("[含上标，公式指数或行标可能未恢复；不能据此确认数值]\n" if superscripts else "")
                + "```text\n" + text + "\n```\n[同页表题/表注]\n" + caption.strip(),
    }


def read_ruled_tables(path):
    """Recover bounded numeric tables; original body warnings remain separate."""
    result = {}
    with pymupdf.open(path) as pdf:
        for number, page in enumerate(pdf, 1):
            regions = recover_ruled_tables(page)
            if regions:
                result[number] = regions
    return result
