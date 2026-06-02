"""Deterministic mock LLM HTTP server.

Impersonates the OpenAI Chat Completions API (``POST /v1/chat/completions``) and
the Anthropic Messages API (``POST /v1/messages``) on localhost so a generated
agent task runs with no real API key and no cost. An incoming prompt is matched
against recorded fixtures (exact -> normalized -> semantic) and the matching
fixture's canned response is returned in the right wire shape. A miss fails
**loudly** (HTTP 500 with the closest fixtures) so a silently-wrong answer can
never masquerade as a real model reply.

Control endpoints (prefixed ``/__``) are for the generation-time gate:

* ``GET  /__calls``    — the call log (count, killed, per-call tier + match).
* ``POST /__reset``    — clear the call log between gate runs.
* ``GET  /__fixtures`` — list loaded fixture ids (+ whether semantic is on).
* ``GET  /healthz``    — liveness.

Run it::

    python -m infra.agent_runtime.mock_llm_server --fixtures path/to/fixtures --port 11434
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .adapters import ANTHROPIC, OPENAI, extract_prompt, render_response
from .call_log import CallLog
from .embedding import try_load_embedder
from .fixtures import Fixture, load_fixtures
from .matching import Embedder, Matcher

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434
DEFAULT_THRESHOLD = 0.9


@dataclass(frozen=True)
class ServeResult:
    status_code: int
    payload: dict[str, Any]


class MockServer:
    """Holds the fixtures, matcher, and call log, and serves one request."""

    def __init__(
        self,
        fixtures: list[Fixture],
        *,
        threshold: float = DEFAULT_THRESHOLD,
        max_calls: int | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.fixtures = fixtures
        self.fixtures_by_id = {fixture.id: fixture for fixture in fixtures}
        self.matcher = Matcher(
            [fixture.as_candidate() for fixture in fixtures],
            threshold=threshold,
            embedder=embedder,
        )
        self.call_log = CallLog(max_calls=max_calls)

    def serve(self, provider: str, endpoint: str, body: dict[str, Any]) -> ServeResult:
        model = str(body.get("model") or "mock")

        if self.call_log.would_exceed():
            self.call_log.record_killed()
            return ServeResult(
                429,
                {
                    "error": {
                        "type": "mock_call_ceiling_exceeded",
                        "code": "call_ceiling",
                        "message": (
                            f"mock call ceiling of {self.call_log.max_calls} exceeded "
                            "— kill switch tripped (likely a runaway agent loop)"
                        ),
                    }
                },
            )

        prompt = extract_prompt(provider, body)
        result = self.matcher.match(prompt)

        if not result.hit:
            self.call_log.record(
                endpoint=endpoint,
                model=model,
                tier=result.tier.value,
                matched_fixture=None,
                similarity=result.similarity,
                prompt=prompt,
            )
            return ServeResult(
                500,
                {
                    "error": {
                        "type": "mock_fixture_miss",
                        "code": "fixture_miss",
                        "message": (
                            "no recorded fixture matched this prompt; refusing to "
                            "fabricate a response. Record a fixture for it or loosen "
                            "the semantic threshold."
                        ),
                        "best_similarity": result.similarity,
                        "closest": [
                            {"fixture": fixture_id, "similarity": similarity}
                            for fixture_id, similarity in result.closest
                        ],
                        "prompt": prompt,
                    }
                },
            )

        fixture = self.fixtures_by_id[result.candidate.key]
        self.call_log.record(
            endpoint=endpoint,
            model=model,
            tier=result.tier.value,
            matched_fixture=fixture.id,
            similarity=result.similarity,
            prompt=prompt,
        )
        return ServeResult(200, render_response(provider, fixture.response, model))


def create_app(server: MockServer) -> FastAPI:
    """Build the FastAPI app bound to *server*."""
    app = FastAPI(title="agent-runtime mock LLM", docs_url=None, redoc_url=None)

    async def _handle(provider: str, endpoint: str, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 — any malformed body is a client error
            return JSONResponse(
                status_code=400,
                content={"error": {"type": "invalid_request", "message": "body is not valid JSON"}},
            )
        result = server.serve(provider, endpoint, body)
        return JSONResponse(status_code=result.status_code, content=result.payload)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:  # noqa: ANN202
        return await _handle(OPENAI, "/v1/chat/completions", request)

    @app.post("/v1/messages")
    async def messages(request: Request) -> JSONResponse:  # noqa: ANN202
        return await _handle(ANTHROPIC, "/v1/messages", request)

    @app.get("/__calls")
    async def calls() -> JSONResponse:  # noqa: ANN202
        return JSONResponse(content=server.call_log.as_dict())

    @app.post("/__reset")
    async def reset() -> JSONResponse:  # noqa: ANN202
        server.call_log.reset()
        return JSONResponse(content={"ok": True})

    @app.get("/__fixtures")
    async def fixtures() -> JSONResponse:  # noqa: ANN202
        return JSONResponse(
            content={
                "count": len(server.fixtures),
                "semantic_enabled": server.matcher.semantic_enabled,
                "threshold": server.matcher.threshold,
                "ids": [fixture.id for fixture in server.fixtures],
            }
        )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:  # noqa: ANN202
        return JSONResponse(content={"status": "ok", "fixtures": len(server.fixtures)})

    return app


def build_server(
    fixtures_dir: str | Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    max_calls: int | None = None,
    use_semantic: bool = True,
) -> MockServer:
    """Load fixtures from disk and assemble a :class:`MockServer`."""
    fixtures = load_fixtures(fixtures_dir)
    embedder = try_load_embedder() if use_semantic else None
    return MockServer(fixtures, threshold=threshold, max_calls=max_calls, embedder=embedder)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic mock LLM server")
    parser.add_argument(
        "--fixtures",
        default=os.getenv("AGENT_MOCK_FIXTURES_DIR", str(Path(__file__).parent / "examples")),
        help="directory of *.json fixtures",
    )
    parser.add_argument("--host", default=os.getenv("AGENT_MOCK_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("AGENT_MOCK_PORT", DEFAULT_PORT)))
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.getenv("AGENT_MOCK_THRESHOLD", DEFAULT_THRESHOLD)),
        help="semantic cosine-similarity match threshold",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=(int(os.getenv("AGENT_MOCK_MAX_CALLS")) if os.getenv("AGENT_MOCK_MAX_CALLS") else None),
        help="call-ceiling kill switch (default: unlimited)",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        default=os.getenv("AGENT_MOCK_SEMANTIC", "1") == "0",
        help="disable the semantic match tier (exact + normalized only)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    args = _parse_args(argv)
    server = build_server(
        args.fixtures,
        threshold=args.threshold,
        max_calls=args.max_calls,
        use_semantic=not args.no_semantic,
    )
    app = create_app(server)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
