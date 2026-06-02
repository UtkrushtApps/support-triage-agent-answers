import json
import httpx
from openai import OpenAI


SYSTEM_PROMPT = "You are a support triage assistant. Use the lookup_customer tool to fetch the customer record before assigning a priority."

PRIORITY_INSTRUCTION = "Based on the customer context below, assign a ticket priority. Reply with exactly one word: low, medium, high, or critical."

LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_customer",
        "description": "Fetch plan tier and incident history for a customer from the CRM.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The unique customer identifier."
                }
            },
            "required": ["customer_id"]
        }
    }
}

REQUIRED_FIELDS = {"customer_id", "plan", "open_incident_count", "last_incident_severity"}


class CustomerLookupError(RuntimeError):
    pass


def lookup_customer(customer_id: str) -> dict:
    """Call the upstream CRM.

    Required fields in a valid response: customer_id, plan,
    open_incident_count, last_incident_severity.
    Returns {} on timeout or unknown id — does not raise.
    """
    try:
        response = httpx.get(
            f"https://crm.internal/customers/{customer_id}",
            timeout=2.0
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}


def _validate_customer_record(customer_id: str, record: dict) -> None:
    """Raise CustomerLookupError when the CRM record is missing or incomplete."""
    if not isinstance(record, dict) or not record:
        raise CustomerLookupError(f"Could not resolve customer_id {customer_id!r}")

    missing_fields = REQUIRED_FIELDS.difference(record)
    if missing_fields:
        raise CustomerLookupError(f"Could not resolve customer_id {customer_id!r}")


def run_agent(client: OpenAI, customer_id: str) -> str:
    """Run the support-triage agent for the given customer_id.

    Returns the priority string (low | medium | high | critical).
    Accepts an injected OpenAI client so tests can supply the mock.
    """
    turn1_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Look up customer {customer_id} and assign a ticket priority."},
    ]

    turn1_response = client.chat.completions.create(
        model="mock",
        messages=turn1_messages,
        tools=[LOOKUP_TOOL],
    )

    tool_call = turn1_response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    looked_up_customer_id = args["customer_id"]
    raw_record = lookup_customer(looked_up_customer_id)
    _validate_customer_record(looked_up_customer_id, raw_record)

    context = json.dumps(raw_record, sort_keys=True)

    turn2_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Customer context: {context}\n{PRIORITY_INSTRUCTION}"},
    ]

    turn2_response = client.chat.completions.create(
        model="mock",
        messages=turn2_messages,
    )

    return turn2_response.choices[0].message.content.strip()
