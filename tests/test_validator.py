import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.latex_compiler import compile_tikz
from src.validator import validate, _check_node_overlaps, _resolve_relative_coords


def test_valid_diagram_passes_visual_validation(tmp_path):
    body = r"""
    \begin{tikzpicture}
      \node[draw, circle] (a) at (0,0) {A};
      \node[draw, circle] (b) at (3,0) {B};
      \draw[->] (a) -- (b);
    \end{tikzpicture}
    """
    result = compile_tikz(body, tmp_path, name="valid")
    assert result.success
    report = validate(result.png_path, body)
    assert not report.has_blocking_issue


def test_overlapping_nodes_are_flagged(tmp_path):
    body = r"""
    \begin{tikzpicture}
      \node (a) at (0,0) {A};
      \node (b) at (0.1,0.1) {B};
    \end{tikzpicture}
    """
    result = compile_tikz(body, tmp_path, name="overlap")
    assert result.success  # compiles fine - this is the point
    report = validate(result.png_path, body)
    kinds = [f.kind for f in report.findings]
    assert "overlap" in kinds


def test_broken_syntax_fails_compilation(tmp_path):
    body = r"""
    \begin{tikzpicture}
      \node (a) at (0,0) {Start}
      \draw (a) -- (2,0)
    \end{tikzpicture}
    """
    result = compile_tikz(body, tmp_path, name="broken")
    assert not result.success
    assert len(result.errors) > 0


def test_circuitikz_pattern_crops_correctly(tmp_path):
    """Regression test for the standalone [tikz,...] vs circuitikz bug
    found during development: the class option must NOT include `tikz`
    or circuitikz diagrams silently render on a full letter page."""
    from src.template_library import LIBRARY

    circuit = next(p for p in LIBRARY if p.key == "circuit")
    result = compile_tikz(circuit.skeleton, tmp_path, name="circuit")
    assert result.success
    report = validate(result.png_path, circuit.skeleton)
    assert not report.has_blocking_issue, report.summary()


# ── Pure-Python tests for relative-positioning overlap detection ─────────────
# These do not require pdflatex and run on every platform.

def test_relative_nodes_flagged_when_too_close():
    """Nodes placed with ``right=of`` at a tiny node distance should be
    flagged just like absolute-coordinate overlaps are."""
    body = r"""
\begin{tikzpicture}[node distance=0.1cm]
  \node (a) at (0,0) {A};
  \node[right=of a] (b) {B};
\end{tikzpicture}
"""
    findings = _check_node_overlaps(body)
    kinds = [f.kind for f in findings]
    assert "overlap" in kinds, (
        "Expected an overlap finding for node distance=0.1cm but got: "
        + str(findings)
    )


def test_relative_nodes_not_flagged_when_well_spaced():
    """Nodes placed with ``right=of`` at a generous node distance (default
    flowchart layout) must NOT produce a false-positive overlap warning."""
    body = r"""
\begin{tikzpicture}[node distance=1.1cm]
  \node (start) at (0,0) {Start};
  \node[below=of start] (step1) {Step 1};
  \node[below=of step1] (end) {End};
\end{tikzpicture}
"""
    findings = _check_node_overlaps(body)
    overlap = [f for f in findings if f.kind == "overlap"]
    assert not overlap, f"Unexpected overlap findings: {overlap}"


def test_relative_node_chain_resolves_correctly():
    """A chain A → B → C (each relative to the previous) must resolve all
    three coordinates and correctly detect only the pair that is too close."""
    # B is 2.0 units right of A, C is 0.1 units right of B → B and C overlap.
    body = r"""
\begin{tikzpicture}[node distance=2.0cm]
  \node (A) at (0,0) {A};
  \node[right=of A] (B) {B};
  \node[right=0.1cm of B] (C) {C};
\end{tikzpicture}
"""
    coords = _resolve_relative_coords(
        body,
        abs_coords={"A": (0.0, 0.0)},
        default_distance=2.0,
    )
    # B should be at x=2.0, C at x=2.1 (explicit 0.1cm override)
    assert abs(coords["B"][0] - 2.0) < 1e-9, f"B.x={coords['B'][0]}"
    assert abs(coords["C"][0] - 2.1) < 1e-9, f"C.x={coords['C'][0]}"

    findings = _check_node_overlaps(body)
    ids_in_findings = {
        name
        for f in findings
        for name in (f.detail.split("'")[1], f.detail.split("'")[3])
    }
    assert "B" in ids_in_findings and "C" in ids_in_findings, (
        f"Expected B/C overlap, got: {findings}"
    )
    # A and B are 2.0 units apart — should NOT be flagged
    ab_finding = any(
        {"A", "B"} <= {f.detail.split("'")[1], f.detail.split("'")[3]}
        for f in findings
    )
    assert not ab_finding, "A and B should not be flagged as overlapping"


def test_unresolvable_anchor_skipped_silently():
    """A relative node whose anchor lives outside the snippet (e.g. in a
    \foreach expansion) must be silently dropped — no exception, no phantom
    finding against an unrelated node."""
    body = r"""
\begin{tikzpicture}[node distance=1.0cm]
  \node (known) at (0,0) {Known};
  \node[right=of phantom] (orphan) {Orphan};
\end{tikzpicture}
"""
    # Should not raise, and should not fabricate an overlap
    findings = _check_node_overlaps(body)
    overlap = [f for f in findings if f.kind == "overlap"]
    assert not overlap, f"No overlap expected for orphan node: {overlap}"
