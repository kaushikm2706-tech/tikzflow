"""
TikZFlow - Agentic AI diagram generator (IBM watsonx.ai Granite backend).

    streamlit run app.py

Problem Statement No.26 - AI-Powered LaTeX Diagram Generator for Academic
Research (IBM SkillsBuild for University Engagements, AICTE-2026).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.agent import TikZAgent  # noqa: E402

st.set_page_config(page_title="TikZFlow", page_icon="📐", layout="wide")

if "agent" not in st.session_state:
    st.session_state.work_dir = Path(tempfile.mkdtemp(prefix="tikzflow_"))
    st.session_state.agent = TikZAgent(work_dir=st.session_state.work_dir)
    st.session_state.result = None

agent: TikZAgent = st.session_state.agent

st.title("📐 TikZFlow")
st.caption(
    "Agentic AI-Powered LaTeX Diagram Generator — describe a diagram in plain English, "
    "get compilable, publication-ready TikZ. Generation is grounded against a verified "
    "pattern library, then self-corrected against real compiler and visual-validation "
    "feedback until it renders cleanly."
)

backend_label = "🟢 IBM watsonx.ai Granite (live)" if agent.backend_name == "watsonx-granite" else "🟡 Local pattern fallback (no cloud credentials configured)"
st.info(f"**Active backend:** {backend_label}")

examples = json.loads((Path(__file__).parent / "examples" / "prompts.json").read_text())

col_input, col_output = st.columns([1, 1.3])

with col_input:
    st.subheader("1. Describe your diagram")
    example_titles = ["— choose an example —"] + [e["title"] for e in examples]
    chosen = st.selectbox("Quick-start examples", example_titles)
    default_prompt = ""
    if chosen != example_titles[0]:
        default_prompt = next(e["prompt"] for e in examples if e["title"] == chosen)

    prompt = st.text_area("Diagram description", value=default_prompt, height=120,
                           placeholder="e.g. Draw a finite automaton with states q0, q1, q2...")

    generate_clicked = st.button("🚀 Generate diagram", type="primary", use_container_width=True)

    if st.session_state.result is not None:
        st.subheader("2. Refine (optional)")
        feedback = st.text_input("Natural-language refinement", placeholder="e.g. make the arrows curved and add a title")
        refine_clicked = st.button("✨ Apply refinement", use_container_width=True)
    else:
        refine_clicked = False
        feedback = ""

if generate_clicked and prompt.strip():
    with st.spinner("Generating → compiling → validating → self-correcting..."):
        st.session_state.result = agent.run(prompt)

if refine_clicked and feedback.strip():
    with st.spinner("Refining diagram..."):
        st.session_state.result = agent.refine(feedback)

with col_output:
    st.subheader("Output")
    result = st.session_state.result
    if result is None:
        st.write("Generate a diagram to see it here.")
    else:
        if result.success and result.final_png:
            st.image(str(result.final_png), caption="Rendered diagram", use_container_width=True)
            st.download_button("⬇ Download .tex", data=result.final_tikz, file_name="diagram.tex")
        else:
            st.error("Could not produce a compiling, visually-valid diagram within the retry budget.")

        with st.expander(f"🔍 Agent trace ({len(result.attempts)} attempt(s))", expanded=not result.success):
            for a in result.attempts:
                status = "✅ accepted" if a.accepted else ("⚠️ compiled, rejected" if a.compile_result.success else "❌ compile failed")
                st.markdown(f"**Attempt {a.attempt_no} — {a.stage}** — {status} — backend: `{a.backend}`")
                if not a.compile_result.success:
                    for err in a.compile_result.errors[:3]:
                        st.code(err.message, language="text")
                elif a.visual_report and a.visual_report.findings:
                    st.code(a.visual_report.summary(), language="text")
                st.code(a.tikz_body, language="latex")

        st.subheader("📊 Token usage this session")
        c1, c2, c3 = st.columns(3)
        c1.metric("Round trips", agent.tokens.round_trips)
        c2.metric("Prompt tokens (est.)", agent.tokens.total_prompt_tokens)
        c3.metric("Completion tokens (est.)", agent.tokens.total_completion_tokens)

st.divider()
st.caption(
    "IBM SkillsBuild for University Engagements · AICTE-2026 · Problem Statement No.26 · "
    "Technology: IBM Watson Studio, IBM Granite Models"
)
