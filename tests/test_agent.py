import pytest
from unittest.mock import patch
from agent import run_agent, CustomerLookupError


VALID_RECORD = {
    "customer_id": "cust_001",
    "plan": "enterprise",
    "open_incident_count": 3,
    "last_incident_severity": "critical",
}


def test_happy_path_returns_priority(llm):
    """When lookup_customer returns a valid record the agent returns the correct priority."""
    with patch("agent.lookup_customer", return_value=VALID_RECORD):
        result = run_agent(llm, "cust_001")
    assert result == "critical"


def test_lookup_failure_raises(llm):
    """When lookup_customer returns {} the agent must raise CustomerLookupError.

    This test is RED on the broken starter (which fabricates an answer instead
    of raising) and GREEN after the candidate applies the fix.
    """
    with patch("agent.lookup_customer", return_value={}):
        with pytest.raises(CustomerLookupError):
            run_agent(llm, "cust_timeout")
