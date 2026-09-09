"""
latex_compiler.py
------------------
Wraps pdflatex to compile a standalone TikZ snippet, parses the .log for
human-readable error messages (line number + message), and rasterizes the
result to PNG for preview. This is the ground-truth "does it actually
compile" check the agent loop depends on.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


TIKZ_LIBS = (
    "arrows.meta,positioning,shapes.geometric,shapes.multipart,"
    "calc,fit,backgrounds,decorations.pathreplacing,decorations.markings,"
    "automata,chains,matrix,patterns"
)

DOCUMENT_TEMPLATE = r"""\documentclass[border=6pt]{{standalone}}
\usepackage{{tikz}}
\usetikzlibrary{{{libs}}}
\usepackage{{circuitikz}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\begin{{document}}
{body}
\end{{document}}
"""


@dataclass
class CompileError:
    line: int | None
    message: str
    raw: str


@dataclass
class CompileResult:
    success: bool
    pdf_path: Path | None
    png_path: Path | None
    log: str
    errors: list[CompileError] = field(default_factory=list)
    tex_source: str = ""


def wrap_tikz(tikz_body: str) -> str:
    """Wrap a raw tikzpicture (or full document) into a compilable standalone doc."""
    stripped = tikz_body.strip()
    if stripped.startswith(r"\documentclass"):
        return stripped
    return DOCUMENT_TEMPLATE.format(libs=TIKZ_LIBS, body=stripped)


_ERROR_LINE_RE = re.compile(r"^l\.(\d+)\s")
_ERROR_MSG_RE = re.compile(r"^! (.+)$", re.MULTILINE)


def _parse_log_errors(log: str) -> list[CompileError]:
    """
    pdflatex's log format interleaves '! <message>' with, a few lines later,
    'l.<N> <code>'. We pair each '!' with the nearest following 'l.N'.
    """
    errors: list[CompileError] = []
    lines = log.splitlines()
    for i, line in enumerate(lines):
        m = _ERROR_MSG_RE.match(line)
        if not m:
            continue
        message = m.group(1)
        line_no = None
        for look_ahead in lines[i : i + 8]:
            lm = _ERROR_LINE_RE.match(look_ahead)
            if lm:
                line_no = int(lm.group(1))
                break
        errors.append(CompileError(line=line_no, message=message, raw=line))
    return errors


def compile_tikz(tikz_body: str, out_dir: Path, name: str = "diagram", dpi: int = 200) -> CompileResult:
    """
    Compile a TikZ snippet end to end: write .tex -> pdflatex -> pdftoppm.
    Never raises on a bad diagram; failures are reported in CompileResult.
    """
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_source = wrap_tikz(tikz_body)
    tex_path = out_dir / f"{name}.tex"
    tex_path.write_text(tex_source, encoding="utf-8")

    try:
        proc = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(out_dir),
                str(tex_path),
            ],
            cwd=out_dir,
            capture_output=True,
            text=True,
            timeout=45,
        )
        log = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired as e:
        log = f"pdflatex timed out: {e}"
        return CompileResult(False, None, None, log, [CompileError(None, "Compilation timed out", log)], tex_source)

    pdf_path = out_dir / f"{name}.pdf"
    success = pdf_path.exists()
    errors = _parse_log_errors(log) if not success else []

    png_path = None
    if success:
        png_prefix = out_dir / name
        try:
            subprocess.run(
                ["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf_path), str(png_prefix)],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
            candidate = out_dir / f"{name}.png"
            if candidate.exists():
                png_path = candidate
        except Exception as e:  # rendering failure shouldn't mask a real compile success
            log += f"\n[pdftoppm warning] {e}"

    return CompileResult(success, pdf_path if success else None, png_path, log, errors, tex_source)


def is_texlive_available() -> bool:
    return shutil.which("pdflatex") is not None
