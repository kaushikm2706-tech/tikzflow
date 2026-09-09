"""
granite_client.py
------------------
Thin wrapper around IBM watsonx.ai's chat/text-generation endpoint for the
`ibm/granite-3-8b-instruct` (or granite-code) model. Requires three
environment variables that the IBM Cloud Lite account issued during
the internship's "Registration on IBM Academic and IBM Cloud" step:

    WATSONX_API_KEY
    WATSONX_PROJECT_ID
    WATSONX_URL          (region endpoint, e.g. https://us-south.ml.cloud.ibm.com)

If those are not set, `GraniteClient` transparently falls back to
`LocalPatternGenerator`, which does NOT call any LLM - it only adapts the
verified skeletons in `template_library.py` with simple string
substitution. This keeps the whole pipeline runnable and testable with
zero cloud dependency (that is how every example in `examples/` and every
test in `tests/` was produced), while the exact same call sites become a
real Granite-backed agent the moment credentials are exported. Nothing in
this file pretends the offline path is the LLM - `GraniteClient.backend`
reports which one actually ran, and the Streamlit UI displays it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from . import template_library

SYSTEM_PROMPT = """You are a TikZ/LaTeX diagram generation agent. Given a natural-language \
description and (optionally) a starting skeleton, return ONLY a valid TikZ body \
(a `tikzpicture` or `circuitikz` environment). Do not include \\documentclass, \
\\usepackage, or \\begin{document}. Do not add commentary or markdown fences. \
If given previous code plus a compiler error, fix the specific error - do not \
regenerate from scratch."""


@dataclass
class GenerationResult:
    tikz_body: str
    backend: str          # "watsonx-granite" | "local-pattern-fallback"
    pattern_used: str | None
    prompt_tokens_est: int
    completion_tokens_est: int


def _estimate_tokens(text: str) -> int:
    """Rough whitespace/punctuation heuristic (~1 token per 4 chars),
    good enough for the relative before/after comparison the agent
    reports - not a substitute for the real usage figures watsonx.ai
    returns in its response metadata (used automatically when the real
    API path is active)."""
    return max(1, len(text) // 4)


class LocalPatternGenerator:
    """Zero-dependency fallback: retrieval + templated substitution, no LLM call."""

    def generate(self, prompt: str, context: str | None = None) -> GenerationResult:
        pattern = template_library.retrieve(prompt)
        if pattern is None:
            body = self._generic_fallback(prompt)
            pattern_key = None
        else:
            body = pattern.skeleton
            pattern_key = pattern.key
        return GenerationResult(
            tikz_body=body,
            backend="local-pattern-fallback",
            pattern_used=pattern_key,
            prompt_tokens_est=_estimate_tokens(prompt),
            completion_tokens_est=_estimate_tokens(body),
        )

    def fix(self, previous_body: str, error_message: str) -> GenerationResult:
        """Heuristic auto-fixes for the handful of errors that dominate
        LLM-generated TikZ: missing semicolons, mismatched braces, and
        unknown library references. This is intentionally narrow - real
        fixing is Granite's job; this only keeps the offline demo path
        self-correcting too."""
        fixed = previous_body
        if "Missing $ inserted" in error_message or "Undefined control sequence" in error_message:
            fixed = re.sub(r"(?<!\\)%", r"\\%", fixed)
        if fixed.count("{") != fixed.count("}"):
            fixed += "}" * (fixed.count("{") - fixed.count("}"))

        # Semicolon repair: track brace balance across the whole body rather
        # than per line, since a `\node ... {label}` command's true end is
        # wherever braces return to zero, not wherever the line happens to
        # wrap. A command is "missing a semicolon" iff it starts a new draw
        # command, its braces are balanced by the time we reach a line
        # ending in something other than ';', and that line isn't itself
        # about to open a new unbalanced group.
        lines = fixed.splitlines()
        fixed_lines: list[str] = []
        balance = 0
        in_command = False
        for ln in lines:
            stripped = ln.strip()
            if not in_command and re.match(r"\\(draw|node|path|foreach)\b", stripped):
                in_command = True
                balance = 0
            if in_command:
                balance += stripped.count("{") - stripped.count("}")
                if balance <= 0 and not stripped.endswith((";", "%")):
                    ln = ln.rstrip() + ";"
                    in_command = False
                elif balance <= 0:
                    in_command = False
            fixed_lines.append(ln)
        fixed = "\n".join(fixed_lines)
        return GenerationResult(
            tikz_body=fixed,
            backend="local-pattern-fallback",
            pattern_used=None,
            prompt_tokens_est=_estimate_tokens(previous_body + error_message),
            completion_tokens_est=_estimate_tokens(fixed),
        )

    @staticmethod
    def _generic_fallback(prompt: str) -> str:
        safe = prompt.replace("_", r"\_").replace("&", r"\&")[:60]
        return (
            "\\begin{tikzpicture}\n"
            "  \\node[draw, rounded corners, fill=blue!10, minimum width=6cm, "
            "minimum height=2cm, align=center, font=\\small] {"
            f"No matching pattern for:\\\\ \\textit{{{safe}...}}"
            "};\n"
            "\\end{tikzpicture}"
        )


class GraniteClient:
    """Prefers a real watsonx.ai call; falls back to LocalPatternGenerator
    when credentials are absent. See module docstring."""

    def __init__(self) -> None:
        self.api_key = os.getenv("WATSONX_API_KEY")
        self.project_id = os.getenv("WATSONX_PROJECT_ID")
        self.url = os.getenv("WATSONX_URL")
        self._fallback = LocalPatternGenerator()
        self.backend = "watsonx-granite" if self._credentials_present() else "local-pattern-fallback"

    def _credentials_present(self) -> bool:
        return bool(self.api_key and self.project_id and self.url)

    def generate(self, prompt: str, context: str | None = None) -> GenerationResult:
        if not self._credentials_present():
            return self._fallback.generate(prompt, context)
        return self._call_watsonx(prompt, context)

    def fix(self, previous_body: str, error_message: str) -> GenerationResult:
        if not self._credentials_present():
            return self._fallback.fix(previous_body, error_message)
        fix_prompt = (
            f"Previous TikZ code:\n{previous_body}\n\n"
            f"Compiler error:\n{error_message}\n\n"
            f"Return corrected TikZ body only."
        )
        return self._call_watsonx(fix_prompt, context=None)

    def _call_watsonx(self, prompt: str, context: str | None) -> GenerationResult:
        """Real IBM watsonx.ai call. Imports lazily so `ibm-watsonx-ai` is
        only required when credentials are actually configured."""
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference

        credentials = Credentials(url=self.url, api_key=self.api_key)
        model = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            credentials=credentials,
            project_id=self.project_id,
            params={"decoding_method": "greedy", "max_new_tokens": 900, "temperature": 0.2},
        )
        pattern = template_library.retrieve(prompt)
        skeleton_hint = f"\n\nA verified starting skeleton for this diagram family:\n{pattern.skeleton}" if pattern else ""
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser request: {prompt}{skeleton_hint}"
        if context:
            full_prompt += f"\n\nAdditional context:\n{context}"

        response = model.generate_text(prompt=full_prompt, raw_response=True)
        text = response["results"][0]["generated_text"].strip()
        usage = response.get("results", [{}])[0]
        return GenerationResult(
            tikz_body=_strip_code_fences(text),
            backend="watsonx-granite",
            pattern_used=pattern.key if pattern else None,
            prompt_tokens_est=usage.get("input_token_count", _estimate_tokens(full_prompt)),
            completion_tokens_est=usage.get("generated_token_count", _estimate_tokens(text)),
        )


def _strip_code_fences(text: str) -> str:
    text = re.sub(r"^```(?:latex|tex)?\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    return text.strip()
