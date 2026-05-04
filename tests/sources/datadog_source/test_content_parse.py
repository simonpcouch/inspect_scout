"""Tests for Datadog content parsing (content_parse.py)."""

import json

import pytest
from inspect_scout.sources._datadog.content_parse import (
    _parse_tool_uses,
    _strip_base64_images,
    _strip_thinking_blocks,
    restructure_anthropic_content,
)


# ---------------------------------------------------------------------------
# _strip_thinking_blocks
# ---------------------------------------------------------------------------


class TestStripThinkingBlocks:
    """Tests for _strip_thinking_blocks."""

    @pytest.mark.parametrize(
        "input_text, expected",
        [
            pytest.param(
                "[thinking signature=abc123]some deep thought[/thinking]",
                "",
                id="single_block",
            ),
            pytest.param(
                "[thinking signature=abc123]line1\nline2\nline3[/thinking]",
                "",
                id="multiline",
            ),
            pytest.param(
                "Hello world",
                "Hello world",
                id="no_block",
            ),
            pytest.param(
                "[thinking signature=abc123]only thinking here[/thinking]",
                "",
                id="only_thinking",
            ),
            pytest.param(
                "[thinking signature=abc]thought[/thinking]\n"
                "[tool_use: my_tool (id: t1)]\n{}\n[/tool_use]",
                "[tool_use: my_tool (id: t1)]\n{}\n[/tool_use]",
                id="thinking_plus_tool_use",
            ),
        ],
    )
    def test_strip(self, input_text: str, expected: str) -> None:
        assert _strip_thinking_blocks(input_text) == expected


# ---------------------------------------------------------------------------
# _strip_base64_images
# ---------------------------------------------------------------------------


class TestStripBase64Images:
    """Tests for _strip_base64_images."""

    def test_base64_replaced(self) -> None:
        blocks = [
            {"type": "image", "source": {"type": "base64", "data": "abc" * 1000}},
        ]
        result = _strip_base64_images(blocks)
        assert result == [{"type": "text", "text": "[image stripped]"}]

    def test_url_image_preserved(self) -> None:
        blocks = [
            {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}},
        ]
        result = _strip_base64_images(blocks)
        assert result == blocks

    def test_no_images_unchanged(self) -> None:
        blocks = [{"type": "text", "text": "hello"}]
        result = _strip_base64_images(blocks)
        assert result == blocks

    def test_empty_list(self) -> None:
        assert _strip_base64_images([]) == []


# ---------------------------------------------------------------------------
# _parse_tool_uses
# ---------------------------------------------------------------------------


class TestParseToolUses:
    """Tests for _parse_tool_uses."""

    def test_single_tool_use(self) -> None:
        content = '[tool_use: my_tool (id: toolu_123)]\n{"key": "val"}\n[/tool_use]'
        result = _parse_tool_uses(content)
        assert result is not None
        assert len(result) == 1
        assert result[0] == {
            "type": "tool_use",
            "id": "toolu_123",
            "name": "my_tool",
            "input": {"key": "val"},
        }

    def test_multiple_tool_uses(self) -> None:
        content = (
            '[tool_use: tool_a (id: id_a)]\n{"a": 1}\n[/tool_use]\n'
            '[tool_use: tool_b (id: id_b)]\n{"b": 2}\n[/tool_use]'
        )
        result = _parse_tool_uses(content)
        assert result is not None
        assert len(result) == 2
        assert result[0]["name"] == "tool_a"
        assert result[1]["name"] == "tool_b"

    def test_tool_use_with_prose(self) -> None:
        content = (
            "I will use a tool now.\n"
            '[tool_use: my_tool (id: toolu_1)]\n{"x": 1}\n[/tool_use]'
        )
        result = _parse_tool_uses(content)
        assert result is not None
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "I will use a tool now."}
        assert result[1]["type"] == "tool_use"

    def test_malformed_json_falls_back_to_empty(self) -> None:
        content = "[tool_use: my_tool (id: toolu_1)]\nnot json\n[/tool_use]"
        result = _parse_tool_uses(content)
        assert result is not None
        assert result[0]["input"] == {}

    def test_no_tool_use_returns_none(self) -> None:
        assert _parse_tool_uses("just some text") is None


# ---------------------------------------------------------------------------
# Integration: restructure_anthropic_content
# ---------------------------------------------------------------------------


class TestRestructureAnthropic:
    """Integration tests for restructure_anthropic_content."""

    def test_tool_use_parsed_for_assistant(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": '[tool_use: search (id: toolu_42)]\n{"q": "hello"}\n[/tool_use]',
            }
        ]
        result = restructure_anthropic_content(messages)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "tool_use"
        assert content[0]["name"] == "search"
        assert content[0]["id"] == "toolu_42"

    def test_tool_use_not_parsed_for_user(self) -> None:
        messages = [
            {
                "role": "user",
                "content": '[tool_use: search (id: toolu_42)]\n{"q": "hello"}\n[/tool_use]',
            }
        ]
        result = restructure_anthropic_content(messages)
        # Should remain a string — tool_use parsing only applies to assistant.
        assert isinstance(result[0]["content"], str)

    def test_thinking_stripped_from_assistant(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": "[thinking signature=sig]deep thought[/thinking]\nHello!",
            }
        ]
        result = restructure_anthropic_content(messages)
        assert result[0]["content"] == "Hello!"

    def test_thinking_not_stripped_from_user(self) -> None:
        content = "[thinking signature=sig]deep thought[/thinking]\nHello!"
        messages = [{"role": "user", "content": content}]
        result = restructure_anthropic_content(messages)
        assert result[0]["content"] == content

    def test_base64_stripped_in_tool_result(self) -> None:
        inner = json.dumps(
            [{"type": "image", "source": {"type": "base64", "data": "x" * 100}}]
        )
        content = f"[tool_result (tool_use_id: tu_1)]{inner}[/tool_result]"
        messages = [{"role": "user", "content": content}]
        result = restructure_anthropic_content(messages)
        tool_block = result[0]["content"][0]
        assert tool_block["type"] == "tool_result"
        assert tool_block["content"] == [{"type": "text", "text": "[image stripped]"}]

    def test_base64_stripped_in_json_list(self) -> None:
        content = json.dumps(
            [
                {"type": "text", "text": "hi"},
                {"type": "image", "source": {"type": "base64", "data": "x" * 100}},
            ]
        )
        messages = [{"role": "assistant", "content": content}]
        result = restructure_anthropic_content(messages)
        blocks = result[0]["content"]
        assert isinstance(blocks, list)
        assert blocks[0] == {"type": "text", "text": "hi"}
        assert blocks[1] == {"type": "text", "text": "[image stripped]"}

    def test_non_string_content_unchanged(self) -> None:
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        result = restructure_anthropic_content(messages)
        assert result[0]["content"] == [{"type": "text", "text": "hi"}]
