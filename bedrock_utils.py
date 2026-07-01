"""
Shared Bedrock helpers used by extract_entities.py, supplier_mapping.py and
item_mapping.py.

Provides:
  - get_bedrock_client(cfg): a bedrock-runtime client whose AWS profile/region come
    from config (env-overridable) and whose timeouts/retries come from the api block.
  - parse_model_json(response): robust extraction of the JSON object from a converse()
    response (does not assume content[0], strips code fences, salvages stray prose).
  - converse_json(client, payload, ...): converse() + parse_model_json with one retry.
"""
from __future__ import annotations
import json
import os
import time
from typing import Any, Dict

import boto3
from botocore.config import Config


class ModelOutputError(RuntimeError):
    """The model response could not be parsed into the expected JSON.

    Deliberately NOT a ValueError: callers surface ValueError messages to the API
    caller, but a parse failure is an internal condition whose message may contain a
    snippet of raw model output, so it must be logged server-side and reported
    generically instead.
    """


# =========================================================
# CLIENT
# =========================================================

def get_bedrock_client(cfg: Dict[str, Any]):
    """
    Build a bedrock-runtime client.

    AWS profile/region: env (AWS_PROFILE / AWS_REGION) overrides cfg["aws"]; if no
    profile is configured, fall back to the default credential chain so it can run in
    environments without a named profile (e.g. an instance role).
    Timeouts/retries come from cfg["api"]["timeout_seconds"] so a stuck call cannot hang
    forever.
    """
    aws = (cfg or {}).get("aws", {}) or {}
    api = (cfg or {}).get("api", {}) or {}
    print(f"AWS: {aws}")

    profile = aws.get("profile") or os.getenv("AWS_PROFILE")
    print(f"PRFILE NAME: {profile}")
    region = (
        aws.get("region")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "us-east-1"
    )
    print(f"REGION: {region}")

    read_timeout = api.get("timeout_seconds", 300)
    max_attempts = int(api.get("max_attempts", 3))

    boto_cfg = Config(
        connect_timeout=10,
        read_timeout=read_timeout,
        retries={"max_attempts": max_attempts, "mode": "adaptive"},
    )

    if profile:
        session = boto3.Session(profile_name=profile)
    else:
        session = boto3.Session()

    return session.client("bedrock-runtime", region_name=region, config=boto_cfg)

# =========================================================
# RESPONSE PARSING
# =========================================================

def _extract_text(response: Dict[str, Any]) -> str:
    """Return the first text block from a converse() response (not assuming index 0)."""
    try:
        content = response["output"]["message"]["content"]
    except (KeyError, TypeError) as e:
        raise ModelOutputError(f"Unexpected Bedrock response shape: {e}")

    texts = [b["text"] for b in content if isinstance(b, dict) and "text" in b]
    if not texts:
        raise ModelOutputError("Bedrock response contained no text block.")
    # Usually a single block; join defensively if the model split it.
    return "\n".join(texts)


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        s = s[nl + 1:] if nl != -1 else s.lstrip("`")
    if s.endswith("```"):
        s = s[: s.rfind("```")]
    return s.strip()


def parse_model_json(response: Dict[str, Any]) -> Any:
    """
    Robustly parse the JSON object returned by a converse() call.

    Handles: text not in the first content block, ```json / ``` fences anywhere, and
    leading/trailing prose (salvages the substring between the first '{' and last '}').
    Raises ValueError with a short preview if no valid JSON can be recovered.
    """
    text = _extract_text(response)
    cleaned = _strip_code_fences(text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ModelOutputError(
        "Model did not return valid JSON. First 200 chars: " + cleaned[:200]
    )


def converse_json(client, payload: Dict[str, Any], retries: int = 1,
                  retry_delay: float = 0.0) -> Any:
    """
    Call client.converse(**payload) and parse the JSON result, retrying once on a parse
    failure (transient bad output). Re-raises the last error if all attempts fail.
    """
    last_err = None
    print(f"AAAAAAAAAA")
    for attempt in range(retries + 1):
        print(f"LLLLL")
        response = client.converse(**payload)
        print(f"KOKOK")
        try:
            return parse_model_json(response)
        except ModelOutputError as e:
            last_err = e
            if attempt < retries:
                if retry_delay:
                    time.sleep(retry_delay)
                continue
            raise
    raise last_err  # pragma: no cover
