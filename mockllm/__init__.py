"""Deterministic mock LLM runtime for agent-engineering assessment tasks.

A generated *agent* task ships a codebase that must call an LLM to run. A real
model is non-deterministic, costs money, and needs API keys — all of which break
the generation-time solvability gate (empty-candidate-fails / reference-passes)
and make candidate grading unfair. This package provides a **deterministic,
keyless, free** stand-in:

* :mod:`infra.agent_runtime.mock_llm_server` — a FastAPI app that impersonates
  the OpenAI Chat Completions API and the Anthropic Messages API on localhost.
* :mod:`infra.agent_runtime.matching` — tiered prompt matching
  (exact -> normalized -> semantic -> loud miss) so legitimate run-to-run prompt
  drift still matches a recorded fixture, while a genuinely different prompt
  fails loudly instead of returning a silent wrong answer.
* :mod:`infra.agent_runtime.fixtures` / :mod:`~.adapters` — the fixture format
  and the provider wire-format translation.
* :mod:`infra.agent_runtime.call_log` — records every call (for gate assertions)
  and enforces a call-ceiling kill-switch (for runaway-loop tasks).

Slice 0 builds and de-risks this in isolation, before any pipeline wiring.
"""
