import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import TikZAgent
from src.granite_client import LocalPatternGenerator


class _BreaksOnceThenFine:
    """Test double: first generation is deliberately broken (all semicolons
    stripped), forcing the agent's real retry loop to exercise its fix path
    instead of just succeeding on attempt one."""

    def __init__(self):
        self.calls = 0
        self._real = LocalPatternGenerator()

    def generate(self, prompt, context=None):
        self.calls += 1
        result = self._real.generate(prompt, context)
        if self.calls == 1:
            result.tikz_body = result.tikz_body.replace(";", "")
        return result

    def fix(self, body, error_message):
        return self._real.fix(body, error_message)


def test_agent_recovers_from_initial_failure(tmp_path):
    agent = TikZAgent(work_dir=tmp_path, max_attempts=4)
    agent.client = _BreaksOnceThenFine()

    result = agent.run("Draw a flowchart with a start, a process step, and an end")

    assert result.success
    assert len(result.attempts) >= 2
    assert result.attempts[0].compile_result.success is False
    assert result.attempts[-1].accepted is True


def test_agent_reports_failure_after_exhausting_retries(tmp_path):
    class _AlwaysBroken:
        def generate(self, prompt, context=None):
            from src.granite_client import GenerationResult
            return GenerationResult(r"\begin{tikzpicture} \node {unclosed ", "test", None, 5, 5)

        def fix(self, body, error_message):
            return self.generate("")

    agent = TikZAgent(work_dir=tmp_path, max_attempts=2)
    agent.client = _AlwaysBroken()
    result = agent.run("anything")

    assert not result.success
    assert len(result.attempts) == 2


def test_token_tracker_accumulates_across_attempts(tmp_path):
    agent = TikZAgent(work_dir=tmp_path, max_attempts=4)
    agent.client = _BreaksOnceThenFine()
    agent.run("Draw a neural network with 2 input nodes and 1 output node")

    assert agent.tokens.round_trips >= 2
    assert agent.tokens.total_tokens > 0
