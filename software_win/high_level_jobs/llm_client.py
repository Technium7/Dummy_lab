from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


def _load_env_files() -> None:
    """Load repo-root .env / .env.local into os.environ (do not override existing)."""
    here = Path(__file__).resolve()
    # software_win/high_level_jobs/llm_client.py -> repo root is parents[2]
    roots = [here.parents[2], here.parents[3] if len(here.parents) > 3 else here.parents[2]]
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        for name in (".env.local", ".env"):
            path = root / name
            if not path.is_file():
                continue
            try:
                for raw in path.read_text(encoding="utf-8").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip("'").strip('"')
                    if key and key not in os.environ:
                        os.environ[key] = val
            except OSError:
                continue


_load_env_files()


class LLMClient(Protocol):
    """Minimal interface: return assistant text (prefer JSON-only)."""

    def complete(self, system: str, user: str) -> str: ...


def _http_post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    # Some hosts (e.g. Groq/Cloudflare) reject urllib's default User-Agent (403/1010).
    hdrs = {"User-Agent": "DummyLab-high_level_jobs/1.0", **headers}
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {err_body[:800]}") from e
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
        "To avoid a local LLM: use `--llm-backend openai` with OPENAI_API_KEY "
        "(and optional OPENAI_BASE_URL), or start the session with `--no-llm`."
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
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        base: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
        }
        model_id = str(self.model or "")
        if "gpt-oss" in model_id:
            # Reasoning models often 400 in json_object mode; keep CoT short.
            base["reasoning_effort"] = "low"

        # Groq JSON-object mode can 400 if the model emits non-JSON (common on gpt-oss).
        attempts: list[dict[str, Any]] = [
            {**base, "temperature": 0, "response_format": {"type": "json_object"}},
            {**base, "temperature": 0},
            dict(base),
        ]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_err: Exception | None = None
        out: dict[str, Any] | None = None
        for payload in attempts:
            try:
                out = _http_post_json(url, headers, payload, self.timeout_sec)
                break
            except RuntimeError as e:
                last_err = e
                msg = str(e)
                if "HTTP 400" in msg or "HTTP 429" in msg:
                    continue
                raise
        if out is None:
            raise last_err if last_err is not None else RuntimeError("chat/completions failed")
        try:
            return str(out["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected chat/completions response: {out!r}") from e
