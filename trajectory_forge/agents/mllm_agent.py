"""MLLM agent: calls a vision-language model via OpenAI-compatible API.

Supports GPT-4o and any OpenAI-compatible endpoint (e.g., Azure, local vLLM).
"""

from __future__ import annotations

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
                    temperature=self.temperature,
                )
                logger.info(f"Response: \n{response.choices[0].message.content}")
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


# ── Single-candidate parsing (backward compatible) ───────────────────────────

def parse_tool_call(response: str) -> tuple[str | None, dict | None]:
    """Parse a <tool_call>...</tool_call> block from the model response.

    Handles two formats:
    1. Simple key: value lines (for scalar tools)
    2. JSON values (for hsl_tool adjustments list)

    Returns
    -------
    (tool_name, params) or (None, None) if parsing fails.
    """
    tool_block = _extract_tag(response, "tool_call")
    if not tool_block:
        return None, None

    lines = [line.strip() for line in tool_block.strip().splitlines() if line.strip()]
    if not lines:
        return None, None

    tool_name = None
    params: dict = {}

    for line in lines:
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()

        if key == "tool":
            tool_name = val
            continue

        # Try JSON parse first (handles lists and nested objects)
        try:
            params[key] = json.loads(val)
        except json.JSONDecodeError:
            # Try numeric parse
            try:
                params[key] = float(val)
            except ValueError:
                params[key] = val  # Keep as string

    if not tool_name:
        return None, None

    return tool_name, params


def parse_thinking(response: str) -> str:
    """Extract the <thinking>...</thinking> block."""
    return _extract_tag(response, "thinking") or ""


def is_stop(response: str) -> bool:
    """Check if the response contains a <stop> signal."""
    return bool(re.search(r"<stop>", response, re.IGNORECASE))


# ── Multi-candidate parsing ──────────────────────────────────────────────────

def parse_multi_tool_calls(
    response: str, num_candidates: int = 3
) -> list[dict]:
    """Parse multiple <candidate_N>...</candidate_N> blocks.

    Each candidate block is expected to contain <thinking> and <tool_call> tags.
    Falls back to single-response parsing if no candidate tags are found.

    Returns
    -------
    list[dict]
        Each dict has keys: cot (str), tool_name (str|None), tool_params (dict|None).
    """
    results = []
    for i in range(1, num_candidates + 1):
        block = _extract_tag(response, f"candidate_{i}")
        if block is None:
            continue
        cot = parse_thinking(block)
        tool_name, tool_params = parse_tool_call(block)
        results.append({
            "cot": cot,
            "tool_name": tool_name,
            "tool_params": tool_params,
        })

    if not results:
        cot = parse_thinking(response)
        tool_name, tool_params = parse_tool_call(response)
        if tool_name is not None and tool_params is not None:
            results.append({
                "cot": cot,
                "tool_name": tool_name,
                "tool_params": tool_params,
            })

    return results


def parse_judge_choice(response: str) -> int | None:
    """Parse <choice>X</choice> from the VLM judge response.

    Falls back to finding the first standalone digit if tags are missing.

    Returns
    -------
    int or None
        The chosen image index, or None if parsing fails.
    """
    match = re.search(r"<choice>\s*(\d+)\s*</choice>", response, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # Fallback: last digit in the response (often the judge just says "2")
    digits = re.findall(r"\b(\d+)\b", response.strip())
    if digits:
        return int(digits[-1])
    return None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_tag(text: str, tag: str) -> str | None:
    """Extract content between <tag>...</tag>."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None
