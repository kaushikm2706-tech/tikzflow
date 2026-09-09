"""
template_library.py
--------------------
Core novelty of TikZFlow: rather than asking Granite to hallucinate TikZ
syntax from scratch every single time (the dominant failure mode of pure
generation - PGF's syntax is dense and LLMs routinely invent nonexistent
keys), we keep a small, hand-verified library of TikZ *patterns* for the
diagram families that dominate academic + technical writing. The agent
retrieves the closest-matching pattern by keyword overlap, and asks
Granite to *adapt* the verified skeleton to the user's specifics (labels,
counts, styling) rather than invent structure. This is retrieval applied
to code synthesis, not to conversation - it trades a little generality
for a large jump in first-try compile rate, which is the actual metric
that matters for an "agentic" pipeline being judged on reliability.

Each entry's `skeleton` is itself a valid, pre-compiled TikZ body (see
tests/test_validator.py), so even the zero-LLM fallback path always
produces something real to look at.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Pattern:
    key: str
    keywords: list[str]
    description: str
    skeleton: str


LIBRARY: list[Pattern] = [
    Pattern(
        key="neural_network",
        keywords=["neural network", "neural net", "layers", "perceptron", "mlp", "input layer", "hidden layer"],
        description="Layered feed-forward neural network with input/hidden/output nodes and connecting edges.",
        skeleton=r"""
\begin{tikzpicture}[
  neuron/.style={circle, draw, minimum size=0.8cm, fill=blue!10},
  layerlabel/.style={font=\small\bfseries}
]
  \foreach \i in {1,...,3}
    \node[neuron] (I-\i) at (0, -\i) {};
  \node[layerlabel] at (0, 0) {Input};

  \foreach \i in {1,...,4}
    \node[neuron, fill=orange!15] (H-\i) at (3, -\i*0.75+0.4) {};
  \node[layerlabel] at (3, 0) {Hidden};

  \foreach \i in {1,...,2}
    \node[neuron, fill=green!15] (O-\i) at (6, -\i-0.5) {};
  \node[layerlabel] at (6, 0) {Output};

  \foreach \i in {1,...,3}
    \foreach \j in {1,...,4}
      \draw[-{Latex[length=1.5mm]}, gray!70] (I-\i) -- (H-\j);
  \foreach \i in {1,...,4}
    \foreach \j in {1,...,2}
      \draw[-{Latex[length=1.5mm]}, gray!70] (H-\i) -- (O-\j);
\end{tikzpicture}
""".strip(),
    ),
    Pattern(
        key="flowchart",
        keywords=["flowchart", "flow chart", "process", "workflow", "decision", "steps"],
        description="Standard flowchart with start/end terminals, process boxes, and a decision diamond.",
        skeleton=r"""
\begin{tikzpicture}[
  node distance=1.1cm,
  start/.style={ellipse, draw, fill=green!15, minimum width=2.2cm, minimum height=0.9cm},
  proc/.style={rectangle, draw, fill=blue!10, minimum width=2.6cm, minimum height=0.9cm, rounded corners=2pt},
  decision/.style={diamond, draw, fill=orange!15, aspect=2, minimum width=2.6cm, inner sep=1pt},
  arr/.style={-{Latex[length=2mm]}}
]
  \node[start] (start) {Start};
  \node[proc, below=of start] (step1) {Process step};
  \node[decision, below=of step1] (dec) {Condition?};
  \node[proc, below=1.6cm of dec] (step2) {Handle result};
  \node[start, below=of step2, fill=red!15] (end) {End};

  \draw[arr] (start) -- (step1);
  \draw[arr] (step1) -- (dec);
  \draw[arr] (dec) -- node[right, font=\footnotesize] {yes} (step2);
  \draw[arr] (step2) -- (end);
  \draw[arr] (dec.west) -- ++(-1.8,0) node[midway, above, font=\footnotesize] {no} |- (step1.west);
\end{tikzpicture}
""".strip(),
    ),
    Pattern(
        key="finite_automaton",
        keywords=["automaton", "automata", "state machine", "fsm", "dfa", "nfa", "states", "transitions"],
        description="Finite state machine using the `automata` TikZ library with labeled transitions.",
        skeleton=r"""
\begin{tikzpicture}[
  ->, >=Stealth, shorten >=1pt, auto, node distance=2.6cm,
  state/.style={circle, draw, minimum size=1cm}
]
  \node[state, initial] (q0) {$q_0$};
  \node[state, right=of q0] (q1) {$q_1$};
  \node[state, accepting, right=of q1] (q2) {$q_2$};

  \path (q0) edge node {0} (q1)
        (q1) edge node {1} (q2)
        (q1) edge [loop above] node {0} (q1)
        (q2) edge [loop above] node {0,1} (q2);
\end{tikzpicture}
""".strip(),
    ),
    Pattern(
        key="sequence_diagram",
        keywords=["sequence diagram", "sequence", "actors", "messages", "api call", "request", "interaction"],
        description="UML-style sequence diagram with lifelines and horizontal message arrows.",
        skeleton=r"""
\begin{tikzpicture}[
  actor/.style={rectangle, draw, fill=blue!10, minimum width=2.2cm, minimum height=0.7cm},
  lifeline/.style={dashed, gray},
  msg/.style={-{Latex[length=2mm]}}
]
  \node[actor] (A) at (0,0) {Client};
  \node[actor] (B) at (5,0) {Server};
  \draw[lifeline] (A.south) -- ++(0,-5);
  \draw[lifeline] (B.south) -- ++(0,-5);

  \draw[msg] (0,-1) -- node[above, font=\footnotesize]{request()} (5,-1);
  \draw[msg] (5,-2.5) -- node[above, font=\footnotesize]{response()} (0,-2.5);
  \draw[msg] (0,-4) -- node[above, font=\footnotesize]{ack()} (5,-4);
\end{tikzpicture}
""".strip(),
    ),
    Pattern(
        key="circuit",
        keywords=["circuit", "resistor", "capacitor", "voltage", "amplifier", "op-amp", "electrical"],
        description="Simple RC circuit built with circuitikz.",
        skeleton=r"""
\begin{circuitikz}[american]
  \draw (0,0) to[V, l=$V_{in}$] (0,3)
        to[R, l=$R_1$] (3,3)
        to[C, l=$C_1$] (3,0)
        -- (0,0);
  \draw (3,3) -- (5,3) to[short, o-o] (5,3) node[right] {$V_{out}$};
\end{circuitikz}
""".strip(),
    ),
    Pattern(
        key="org_chart",
        keywords=["org chart", "organization", "hierarchy", "tree", "architecture layers", "system architecture"],
        description="Hierarchical box tree - system architecture or org-chart style.",
        skeleton=r"""
\begin{tikzpicture}[
  box/.style={rectangle, draw, fill=blue!10, rounded corners=2pt, minimum width=3cm, minimum height=0.8cm, align=center},
  level distance=1.6cm,
  level 1/.style={sibling distance=3.4cm},
  level 2/.style={sibling distance=3.4cm},
  edge from parent/.style={draw, -{Latex[length=2mm]}}
]
  \node[box] (root) {Top Level}
    child { node[box] (a) {Component A} }
    child { node[box] (b) {Component B}
      child { node[box, fill=orange!15] (b1) {Sub B.1} }
      child { node[box, fill=orange!15] (b2) {Sub B.2} }
    };
\end{tikzpicture}
""".strip(),
    ),
]


def retrieve(prompt: str) -> Pattern | None:
    """Keyword-overlap retrieval. Returns the best-matching pattern, or None
    if nothing scores above a minimal relevance bar (falls through to pure
    generation)."""
    prompt_lower = prompt.lower()
    best: tuple[int, Pattern] | None = None
    for pattern in LIBRARY:
        score = sum(1 for kw in pattern.keywords if kw in prompt_lower)
        if score and (best is None or score > best[0]):
            best = (score, pattern)
    return best[1] if best else None
