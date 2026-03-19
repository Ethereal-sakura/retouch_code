"""MLLM agent: calls a vision-language model via OpenAI-compatible API.

Supports GPT-4o and any OpenAI-compatible endpoint (e.g., Azure, local vLLM).
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import time
from typing import Any

logger = logging.getLogger(__name__)


class MLLMAgent:
    """Thin wrapper around the OpenAI chat completions API with vision support."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        request_timeout: int = 60,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": request_timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def call(
        self,
        system_prompt: str,
        messages: list[dict],
        max_retries: int = 3,
        retry_delay: float = 5.0,
        temperature: float | None = None,
    ) -> str:
        """Call the MLLM with system prompt and message history.

        Parameters
        ----------
        system_prompt : str
            The system prompt string.
        messages : list[dict]
            List of message dicts with 'role' and 'content' keys.
            Content may be a string or list of content blocks (text/image_url).
        max_retries : int
            Number of retries on transient errors.
        retry_delay : float
            Seconds to wait between retries.

        Returns
        -------
        str
            The model's response text.
        """
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature if temperature is None else float(temperature),
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"API call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    raise


def extract_json_object(response: str) -> dict[str, Any] | None:
    """Extract the first valid JSON-like object from a model response."""
    response = response.strip()
    if not response:
        return None

    candidates = [response]
    if "```" in response:
        fenced = re.findall(r"```(?:json)?\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
        candidates = fenced + candidates

    first = response.find("{")
    last = response.rfind("}")
    if first != -1 and last != -1 and first < last:
        candidates.append(response[first:last + 1])

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        parsed = _parse_json_like(candidate)
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_planner_response(response: str) -> dict[str, Any] | None:
    """Parse the planner's JSON response into a normalized dictionary."""
    payload = extract_json_object(response)
    if not payload:
        return None

    candidates = []
    raw_candidates = payload.get("candidates", [])
    if isinstance(raw_candidates, list):
        for item in raw_candidates:
            if not isinstance(item, dict):
                continue
            parameters = item.get("parameters", {})
            if not isinstance(parameters, dict):
                parameters = {}
            candidates.append(
                {
                    "tool": str(item.get("tool", "")).strip(),
                    "parameters": parameters,
                    "reason": str(item.get("reason", "")).strip(),
                }
            )

    return {
        "main_issue": str(payload.get("main_issue", "")).strip().lower(),
        "candidates": candidates,
    }


def parse_thinking(response: str) -> str:
    """Extract the <thinking>...</thinking> block."""
    return _extract_tag(response, "thinking") or ""


def _extract_tag(text: str, tag: str) -> str | None:
    """Extract content between <tag>...</tag>."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def _parse_json_like(candidate: str) -> dict[str, Any] | None:
    """Parse slightly-invalid JSON that VLMs commonly emit."""
    parsers = (
        lambda text: json.loads(text),
        lambda text: json.loads(_sanitize_json_like(text)),
        lambda text: ast.literal_eval(_pythonize_json_literals(_sanitize_json_like(text))),
    )
    for parser in parsers:
        try:
            parsed = parser(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _sanitize_json_like(text: str) -> str:
    text = text.strip()
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    text = re.sub(r":\s*\+(\d+(?:\.\d+)?)", r": \1", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _pythonize_json_literals(text: str) -> str:
    text = re.sub(r"\btrue\b", "True", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfalse\b", "False", text, flags=re.IGNORECASE)
    text = re.sub(r"\bnull\b", "None", text, flags=re.IGNORECASE)
    return text
