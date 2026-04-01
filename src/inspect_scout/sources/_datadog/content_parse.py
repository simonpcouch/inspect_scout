"""Parse pre-stringified Datadog content back into structured blocks.

The Datadog LLM Observability Export API delivers message content as
pre-stringified text. Tool results with images arrive as flat strings like::

    [tool_result (tool_use_id: toolu_...)] [...JSON...] [/tool_result]

This module parses those strings back into structured Anthropic content
blocks so that ``inspect_ai``'s ``messages_from_anthropic`` can produce
proper ``ChatMessageTool``, ``ContentImage``, etc.
"""

import json
import re
from logging import getLogger
from typing import Any

logger = getLogger(__name__)

# Matches [tool_result (tool_use_id: ID)] ... [/tool_result] blocks.
# Uses DOTALL so `.` matches newlines within the block body.
_TOOL_RESULT_RE = re.compile(
    r"\[tool_result\s*\(tool_use_id:\s*([^)]+)\)\]"
    r"(.*?)"
    r"\[/tool_result\]",
    re.DOTALL,
)


def restructure_anthropic_content(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Parse stringified Datadog content into structured Anthropic blocks.

    Processes each message's ``content`` field:

    1. **Tool-result wrappers** — ``[tool_result ...]...[/tool_result]``
       are parsed into ``{"type": "tool_result", ...}`` blocks.
    2. **Raw JSON list** — if no wrappers found, tries ``json.loads``;
       if result is a list of dicts each with a ``"type"`` key, uses it.
    3. **Fallthrough** — returns original string unchanged.

    Args:
        messages: List of Anthropic-style ``{role, content}`` dicts.

    Returns:
        Same list with ``content`` fields restructured where possible.
        Messages are copied (not mutated in place).
    """
    result = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, str):
            result.append(msg)
            continue

        parsed = _parse_tool_results(content)
        if parsed is not None:
            result.append({**msg, "content": parsed})
            continue

        parsed = _try_json_list(content)
        if parsed is not None:
            result.append({**msg, "content": parsed})
            continue

        result.append(msg)

    return result


def _parse_tool_results(content: str) -> list[dict[str, Any]] | None:
    """Parse ``[tool_result ...]`` wrappers into structured blocks.

    Returns ``None`` if no tool_result wrappers are found.
    """
    matches = list(_TOOL_RESULT_RE.finditer(content))
    if not matches:
        return None

    blocks: list[dict[str, Any]] = []
    last_end = 0

    for match in matches:
        # Capture any text between the previous block and this one.
        between = content[last_end : match.start()].strip()
        if between:
            blocks.append({"type": "text", "text": between})

        tool_use_id = match.group(1).strip()
        body = match.group(2).strip()

        inner = _parse_inner_content(body)

        blocks.append(
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": inner,
            }
        )
        last_end = match.end()

    # Capture any trailing text after the last tool_result block.
    trailing = content[last_end:].strip()
    if trailing:
        blocks.append({"type": "text", "text": trailing})

    return blocks


def _parse_inner_content(body: str) -> list[dict[str, Any]]:
    """Parse the inner content of a tool_result block.

    Tries JSON list first, then wraps as a single text block.
    """
    parsed = _try_json_list(body)
    if parsed is not None:
        return parsed

    # Try parsing as a single JSON object.
    try:
        obj = json.loads(body)
        if isinstance(obj, dict) and "type" in obj:
            return [obj]
    except (json.JSONDecodeError, TypeError):
        pass

    if body:
        return [{"type": "text", "text": body}]
    return []


def _try_json_list(content: str) -> list[dict[str, Any]] | None:
    """Try to parse content as a JSON list of typed blocks.

    Returns ``None`` if parsing fails or result isn't a list of dicts
    with ``"type"`` keys.
    """
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None

    if (
        isinstance(parsed, list)
        and parsed
        and all(isinstance(item, dict) and "type" in item for item in parsed)
    ):
        return parsed

    logger.debug("JSON parsed but not a list of typed blocks: %s", type(parsed).__name__)
    return None
