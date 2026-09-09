"""
validator.py
------------
The "Compile-Validate" agent's second half. A TikZ file can compile
perfectly and still be *wrong*: blank output (coordinates off-canvas),
node labels stacked on top of each other, or a canvas so large the
content is a speck in the corner. pdflatex's exit code says nothing
about any of this.

This module inspects the rendered PNG directly (pixel statistics) plus a
lightweight coordinate-level parse of simple `\node (id) at (x,y) {...}`
declarations, and returns concrete, re-promptable findings the agent can
feed back to the generator - the same "visual verification" gap flagged
in recent TikZ-generation literature (imperfect visual verification is
the dominant remaining failure mode once compilation succeeds).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

NODE_COORD_RE = re.compile(
    r"\\node\s*(?:\[[^\]]*\])?\s*\(([\w:-]+)\)\s*at\s*\(([\-\d.]+)\s*,\s*([\-\d.]+)\)"
)

# Matches the `node distance=<val>` key inside a tikzpicture option list,
# e.g.  \begin{tikzpicture}[node distance=1.1cm, ...]
_NODE_DISTANCE_RE = re.compile(r"node\s+distance\s*=\s*([\d.]+)")

# Matches relative placement nodes:
#   \node[<opts>] (id) {label};   where opts contains  above=of REF, right=of REF, etc.
# Captured groups: (1) full option string, (2) new node id, (3) label (ignored)
_REL_NODE_RE = re.compile(
    r"\\node\s*\[([^\]]*)\]\s*\(([\w:-]+)\)\s*\{"
)

# Direction vectors (unit distance) for each positioning keyword.
_DIR: dict[str, tuple[float, float]] = {
    "right":       ( 1.0,  0.0),
    "left":        (-1.0,  0.0),
    "above":       ( 0.0,  1.0),
    "below":       ( 0.0, -1.0),
    "above right": ( 1.0,  1.0),
    "above left":  (-1.0,  1.0),
    "below right": ( 1.0, -1.0),
    "below left":  (-1.0, -1.0),
}

# Parses  "above=1.5cm of ref"  or  "right=of ref"  (distance optional)
_REL_SPEC_RE = re.compile(
    r"(above\s+right|above\s+left|below\s+right|below\s+left|above|below|right|left)"
    r"\s*=\s*(?:([\d.]+)(?:cm|pt|mm|em|ex|in)?\s+of\s+|(of\s+))"
    r"([\w:-]+)"
)


@dataclass
class VisualFinding:
    kind: str          # "blank", "near_edge", "overlap", "ok"
    detail: str
    severity: str = "warning"   # "warning" | "error"


@dataclass
class VisualReport:
    findings: list[VisualFinding] = field(default_factory=list)

    @property
    def has_blocking_issue(self) -> bool:
        return any(f.severity == "error" for f in self.findings)

    def summary(self) -> str:
        if not self.findings:
            return "Visual validation passed: non-blank render, no detected node collisions."
        return "\n".join(f"- [{f.severity.upper()}] {f.kind}: {f.detail}" for f in self.findings)


def _check_blank_canvas(png_path: Path, threshold: float = 0.995) -> VisualFinding | None:
    """Flag a render that is >threshold fraction pure background (white/transparent)."""
    img = Image.open(png_path).convert("L")
    pixels = list(img.getdata())
    if not pixels:
        return VisualFinding("blank", "Rendered image has zero pixels.", "error")
    white_like = sum(1 for p in pixels if p > 250)
    frac = white_like / len(pixels)
    if frac >= threshold:
        return VisualFinding(
            "blank",
            f"{frac*100:.1f}% of the canvas is blank - likely off-canvas coordinates "
            f"or an empty tikzpicture body.",
            "error",
        )
    return None


def _resolve_relative_coords(
    tikz_source: str,
    abs_coords: dict[str, tuple[float, float]],
    default_distance: float,
) -> dict[str, tuple[float, float]]:
    """
    Walk through every relative-positioned node declaration in *tikz_source*
    (``\\node[right=of X] (Y) {...}``) and compute an approximate (x, y) for
    each one by stepping `default_distance` units in the stated direction from
    the anchor node's position.  Nodes that reference an anchor not yet seen
    are queued and retried until no more progress can be made (handles chains
    like A→B→C where B depends on A and C depends on B).

    The result dict merges the already-resolved *abs_coords* with any newly
    derived positions.  Nodes that remain unresolvable (anchor outside the
    file) are omitted silently — better to miss a check than to fabricate
    coordinates.
    """
    coords: dict[str, tuple[float, float]] = dict(abs_coords)
    pending: list[tuple[str, str, tuple[float, float], float]] = []  # (new_id, anchor, dir_vec, dist)

    for match in _REL_NODE_RE.finditer(tikz_source):
        opts, new_id = match.group(1), match.group(2)
        if new_id in coords:
            continue  # already resolved via absolute placement
        rel = _REL_SPEC_RE.search(opts)
        if rel is None:
            continue
        direction_key = re.sub(r"\s+", " ", rel.group(1).strip())
        explicit_dist = float(rel.group(2)) if rel.group(2) else None
        anchor_id = rel.group(4)
        dir_vec = _DIR.get(direction_key)
        if dir_vec is None:
            continue
        dist = explicit_dist if explicit_dist is not None else default_distance
        pending.append((new_id, anchor_id, dir_vec, dist))

    # Resolve in passes until the queue is exhausted or we make no progress.
    max_passes = len(pending) + 1
    for _ in range(max_passes):
        if not pending:
            break
        still_pending = []
        for new_id, anchor_id, dir_vec, dist in pending:
            if anchor_id in coords:
                ax, ay = coords[anchor_id]
                coords[new_id] = (ax + dir_vec[0] * dist, ay + dir_vec[1] * dist)
            else:
                still_pending.append((new_id, anchor_id, dir_vec, dist))
        if len(still_pending) == len(pending):
            break  # no progress — remaining anchors are external
        pending = still_pending

    return coords


def _check_node_overlaps(tikz_source: str, min_distance: float = 0.55) -> list[VisualFinding]:
    """
    Flag any pair of nodes whose centers sit closer than `min_distance` TikZ
    units — a strong signal of overlapping labels/shapes.

    Covers both absolute-coordinate nodes (``\\node (id) at (x,y)``) and
    relative-positioned nodes (``\\node[right=of X] (Y)``).  For the latter
    the implied coordinate is derived from the anchor node's position plus one
    `node distance` step in the stated direction, using the ``node distance``
    value declared in the tikzpicture preamble (defaulting to 1.0 if absent).
    Nodes whose anchor cannot be resolved within the file are silently skipped
    rather than emitting a spurious finding.
    """
    # 1. Harvest all absolute-coordinate nodes.
    abs_coords: dict[str, tuple[float, float]] = {}
    for match in NODE_COORD_RE.finditer(tikz_source):
        node_id = match.group(1)
        abs_coords[node_id] = (float(match.group(2)), float(match.group(3)))

    # 2. Extract `node distance` from the tikzpicture option block (first match).
    nd_match = _NODE_DISTANCE_RE.search(tikz_source)
    node_distance = float(nd_match.group(1)) if nd_match else 1.0

    # 3. Resolve relative-positioned nodes, extending the coord map.
    all_coords = _resolve_relative_coords(tikz_source, abs_coords, node_distance)

    # 4. Pairwise distance check across the full resolved set.
    coord_list = list(all_coords.items())   # [(id, (x, y)), ...]
    findings: list[VisualFinding] = []
    for i in range(len(coord_list)):
        for j in range(i + 1, len(coord_list)):
            id_a, (xa, ya) = coord_list[i]
            id_b, (xb, yb) = coord_list[j]
            dist = ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5
            if dist < min_distance:
                findings.append(
                    VisualFinding(
                        "overlap",
                        f"Nodes '{id_a}' and '{id_b}' are only {dist:.2f} units apart "
                        f"(< {min_distance}) - labels will likely collide.",
                        "warning",
                    )
                )
    return findings


def validate(png_path: Path | None, tikz_source: str) -> VisualReport:
    findings: list[VisualFinding] = []

    if png_path is None or not png_path.exists():
        findings.append(VisualFinding("blank", "No render was produced to inspect.", "error"))
        return VisualReport(findings)

    blank = _check_blank_canvas(png_path)
    if blank:
        findings.append(blank)

    findings.extend(_check_node_overlaps(tikz_source))

    return VisualReport(findings)
