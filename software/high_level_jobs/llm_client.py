from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class LLMClient(Protocol):
    """Minimal interface: return assistant text (prefer JSON-only)."""

    def complete(self, system: str, user: str) -> str: ...


def _http_post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _ollama_unreachable_hint(exc: BaseException) -> str:
    """Extra context when the TCP connection to Ollama is refused or unreachable."""
    text = f"{type(exc).__name__}: {exc}".lower()
    refused = (
        "10061" in str(exc)
        or "actively refused" in text
        or "connection refused" in text
        or "errno 111" in text
        or "econnrefused" in text
    )
    if not refused:
        return (
            "Check that the Ollama app is running or `ollama serve` is active, "
            "and that OLLAMA_HOST matches where it listens."
        )

    return (
        "Nothing is listening on the Ollama HTTP port (default 11434). "
        "On Windows: open the Ollama app from the Start menu so the server starts, "
        "or run `ollama serve` in a terminal. "
        "Then pull a model once: `ollama pull llama3.2` (or set OLLAMA_MODEL). "
        "If Ollama runs on another machine, set OLLAMA_HOST to that base URL. "
        "To avoid a local LLM: use `--backend openai` with OPENAI_API_KEY (and optional OPENAI_BASE_URL), "
        "or run a hand-written job with `python -m high_level_jobs run high_level_jobs/jobs/examples/go_home_robot.yaml --execute`."
    )


@dataclass
class OllamaClient:
    """
    Local LLM via Ollama (no API key). Install: https://ollama.com
    Set OLLAMA_HOST (default http://127.0.0.1:11434) and OLLAMA_MODEL.
    Use `format: json` in the API to nudge valid JSON output.
    """

    host: str = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    model: str = os.environ.get("OLLAMA_MODEL", "llama3.2")
    timeout_sec: float = 120.0

    def __post_init__(self) -> None:
        if not (self.model or "").strip():
            self.model = os.environ.get("OLLAMA_MODEL", "llama3.2")

    def complete(self, system: str, user: str) -> str:
        url = f"{self.host.rstrip('/')}/api/chat"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
        }
        try:
            out = _http_post_json(url, {"Content-Type": "application/json"}, payload, self.timeout_sec)
        except urllib.error.URLError as e:
            hint = _ollama_unreachable_hint(e)
            raise RuntimeError(
                f"Ollama request failed ({url}).\n{hint}\nOriginal error: {e}"
            ) from e
        msg = out.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected Ollama response shape: {out!r}")
        return content


@dataclass
class OpenAICompatClient:
    """
    Hosted or local OpenAI-compatible HTTP API (v1/chat/completions).
    Examples: OpenAI, Groq, Together, vLLM, LM Studio (enable server).
    Env: OPENAI_API_KEY, OPENAI_BASE_URL (default https://api.openai.com/v1), OPENAI_MODEL.
    """

    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    timeout_sec: float = 120.0

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (self.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = self.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def complete(self, system: str, user: str) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set for OpenAICompatClient")
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        out = _http_post_json(url, headers, payload, self.timeout_sec)
        try:
            return str(out["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected chat/completions response: {out!r}") from e
