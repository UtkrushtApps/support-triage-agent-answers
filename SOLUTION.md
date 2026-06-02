# Solution Steps

1. Inspect the agent flow in `agent.py` and identify the failure mode: `lookup_customer()` catches upstream errors and returns `{}` instead of raising, while `run_agent()` currently passes that raw result directly into TURN 2.

2. Keep the two-turn structure unchanged. The fix belongs immediately after the local tool call in `run_agent()`, before serializing the record into the TURN 2 prompt.

3. Add a small validation helper in `agent.py` that accepts the `customer_id` and returned record, then checks whether the record is a non-empty dict and contains every required field: `customer_id`, `plan`, `open_incident_count`, and `last_incident_severity`.

4. If validation fails, raise the already-defined `CustomerLookupError` with a message that clearly includes the unresolved `customer_id`. This ensures silent CRM failures are surfaced to callers.

5. Call that validation helper right after `raw_record = lookup_customer(...)`. Do not build `context`, do not construct TURN 2 messages, and do not make the second LLM call if validation fails.

6. Preserve happy-path behavior exactly: when the record is valid, continue to serialize it with `json.dumps(..., sort_keys=True)`, send TURN 2 as before, and return the stripped priority string from the model response.

7. Leave `main.py` behavior intact so CLI runs still print the returned priority on success and print the lookup error on failure.

8. Run the tests. `test_lookup_failure_raises` should now pass because `CustomerLookupError` is raised on `{}`, while `test_happy_path_returns_priority` should remain green because valid records still follow the original execution path.

