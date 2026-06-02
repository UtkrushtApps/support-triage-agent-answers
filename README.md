# Support Triage Agent

## Task Overview

This repository contains a small LLM-powered support-triage agent that assigns a priority level (`low`, `medium`, `high`, or `critical`) to incoming support tickets. For each ticket the agent first calls a `lookup_customer` tool to retrieve the customer's plan tier and incident history from an internal CRM, then asks the model to produce a prioritisation decision based on that context. The agent runs against a deterministic local mock LLM — no real model or network calls are made during tests.

The agent contains a planted reliability bug: the `lookup_customer` tool can silently fail by returning an empty dict `{}` instead of raising an exception, and the agent currently trusts that empty result and passes it to the model. The model, receiving no real customer context, confidently fabricates a priority — a critical enterprise customer whose lookup timed out can be silently triaged as `low` with no signal that anything went wrong.

Your job is to find where the silent failure is being ignored, surface it with a structured exception, and confirm that the happy path continues to work correctly.

## Objectives

- Identify the exact point in `agent.py` where the silent tool failure is trusted without validation.
- Raise `CustomerLookupError` (already defined in `agent.py`) when `lookup_customer` returns an empty or incomplete record, before the agent proceeds to ask the model for a priority.
- Ensure that a valid customer record still flows through to the model and produces the correct priority string.
- Make the test `test_lookup_failure_raises` pass (it is intentionally red against the starter code).

## How to Verify

**Run the test suite:**

```bash
pip install -r requirements.txt
pytest tests/ -v
```

After your fix, all tests — including `test_lookup_failure_raises` — should pass.

**Run the agent manually against the mock:**

In one terminal, start the mock LLM server:

```bash
python -m mockllm.mock_llm_server --fixtures fixtures
```

In a second terminal, run the agent:

```bash
# Happy path — valid customer record
LLM_BASE_URL=http://127.0.0.1:11434 python main.py cust_001

# Failure path — lookup returns {}
LLM_BASE_URL=http://127.0.0.1:11434 python main.py cust_timeout
```

Before your fix, both commands return a priority string. After your fix, the second command surfaces a `CustomerLookupError`.

## Helpful Tips

- Consider what the agent does with the value returned by `lookup_customer` before it constructs the second set of messages — that is where the trust decision happens.
- Think about what distinguishes a valid customer record from a failed one; the required fields are documented in the tool's return-type comment.
- Explore the existing fixtures in `fixtures/` to understand exactly what the mock returns for each input and how the broken agent reaches TURN 2 with empty content.
- Review the `CustomerLookupError` class already defined in `agent.py` — it is there for a reason.
