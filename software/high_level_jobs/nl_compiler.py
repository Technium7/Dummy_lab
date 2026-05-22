from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from high_level_jobs.llm_client import LLMClient
from high_level_jobs.schema import JobSpec, load_job_yaml


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_json_object(text: str) -> str:
    """Allow models to wrap JSON in fences; strip and take first {...} block."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL)
    if fence:
        return fence.group(1)
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        return t[start : end + 1]
    return t


def compile_natural_language(
    human_text: str,
    client: LLMClient,
    *,
    system_prompt: str,
    repair_attempts: int = 2,
) -> JobSpec:
    """
    Ask the LLM to emit job YAML/JSON matching the same schema as files in jobs/examples/.
    Parse via load_job_yaml (YAML is a superset of JSON).
    """
    base_user = (
        "Convert the following human command into ONE job document.\n"
        "Output rules: JSON only, no markdown, no commentary.\n"
        "The JSON must be parseable as YAML and must validate against the schema "
        "described in the system message.\n\n"
        f"HUMAN COMMAND:\n{human_text}\n"
    )

    last_err: Exception | None = None
    user = base_user
    for attempt in range(repair_attempts + 1):
        raw = client.complete(system_prompt, user)
        blob = _extract_json_object(raw)
        try:
            # JSON is valid YAML for our purposes
            spec = load_job_yaml(blob)
            return spec
        except Exception as e:
            last_err = e
            user = (
                base_user
                + f"\nYour previous output failed validation/parsing:\n{blob}\n\nError: {e}\n"
                + "Reply again with corrected JSON only."
            )
    assert last_err is not None
    raise last_err


def save_job_yaml(spec: JobSpec, path: Path) -> None:
    from high_level_jobs.schema import job_to_yaml_dict

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(job_to_yaml_dict(spec), sort_keys=False), encoding="utf-8")
