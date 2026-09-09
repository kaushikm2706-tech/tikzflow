"""
token_tracker.py
-----------------
Accumulates prompt/completion token counts across an agent run so the UI
(and the "Token Usage - Before/After Prompt" submission slide) can show a
concrete before/after comparison: naive one-shot generation vs. the
retrieval-grounded, self-correcting pipeline. The pattern-library approach
is expected to *reduce* completion tokens (less code to generate from
scratch) while cutting the number of round trips needed to reach a
compiling diagram (fewer retries = fewer prompt tokens spent on error
messages).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TokenTracker:
    events: list[dict] = field(default_factory=list)

    def log(self, stage: str, prompt_tokens: int, completion_tokens: int, backend: str) -> None:
        self.events.append(
            {
                "stage": stage,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "backend": backend,
            }
        )

    @property
    def total_prompt_tokens(self) -> int:
        return sum(e["prompt_tokens"] for e in self.events)

    @property
    def total_completion_tokens(self) -> int:
        return sum(e["completion_tokens"] for e in self.events)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def round_trips(self) -> int:
        return len(self.events)

    def reset(self) -> None:
        self.events.clear()
