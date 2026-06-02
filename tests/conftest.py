"""Pytest fixtures injected verbatim into every agent-task repo (as tests/conftest.py).

Boots the vendored mock LLM (the `mockllm` package, copied from
infra.agent_runtime) as a real localhost server, and hands tests a real `openai`
client pointed at it. Hermetic + deterministic: no real model, no API key, no
network. This is exactly how the candidate runs the agent manually, so tests and
manual runs share one code path.

Fixtures live in ``<repo>/fixtures``; the agent under test is imported by the
test modules directly (e.g. ``from agent import run_agent``).
"""
import socket
import threading
import time
from pathlib import Path

import httpx
import openai
import pytest
import uvicorn

from mockllm.mock_llm_server import build_server, create_app

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="session")
def server_url():
    """Boot the mock LLM on an ephemeral localhost port for the test session."""
    server = build_server(FIXTURES_DIR, use_semantic=False)
    app = create_app(server)
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            if httpx.get(base + "/healthz", timeout=0.5).status_code == 200:
                break
        except Exception:
            time.sleep(0.05)
    else:
        raise RuntimeError("mock LLM server did not start")
    yield base
    srv.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def llm(server_url):
    """A real openai client wired to the mock; call log reset per test."""
    httpx.post(server_url + "/__reset")
    return openai.OpenAI(api_key="mock", base_url=server_url + "/v1")
