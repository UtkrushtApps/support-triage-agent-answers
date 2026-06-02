import os
import sys
from openai import OpenAI
from agent import run_agent, CustomerLookupError


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py <customer_id>")
        sys.exit(1)

    customer_id = sys.argv[1]
    base_url = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434").rstrip("/") + "/v1"

    client = OpenAI(base_url=base_url, api_key="mock")

    try:
        priority = run_agent(client, customer_id)
        print(f"Ticket priority for customer {customer_id!r}: {priority}")
    except CustomerLookupError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
