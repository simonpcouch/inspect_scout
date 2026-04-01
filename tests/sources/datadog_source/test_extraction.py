"""Tests for Datadog message/output/token extraction."""

import json

import pytest
from inspect_scout.sources._datadog.content_parse import restructure_anthropic_content
from inspect_scout.sources._datadog.detection import Provider
from inspect_scout.sources._datadog.extraction import (
    _extract_system_text,
    _extract_tool_calls,
    _normalize_messages,
    _parse_tool_schema,
    _simple_message_conversion,
    extract_input_messages,
    extract_output,
    extract_tools,
    extract_usage,
    sum_tokens,
)

from .mocks import create_llm_span

pytestmark = pytest.mark.usefixtures("no_fallback_warnings")


class TestExtractInputMessages:
    """Tests for extract_input_messages function."""

    @pytest.mark.asyncio
    async def test_extract_from_messages(self) -> None:
        """Extract messages from meta.input.messages."""
        span = create_llm_span(
            input_messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        )
        messages = await extract_input_messages(span, Provider.OPENAI)

        assert len(messages) >= 1
        assert any(m.role == "user" for m in messages)

    @pytest.mark.asyncio
    async def test_extract_from_value_fallback(self) -> None:
        """Fall back to input.value when no messages."""
        span = create_llm_span()
        span["input"] = {"value": "What is 2+2?"}
        messages = await extract_input_messages(span, Provider.OPENAI)

        assert len(messages) == 1
        assert messages[0].role == "user"
        assert "2+2" in messages[0].content

    @pytest.mark.asyncio
    async def test_extract_empty_input(self) -> None:
        """Return empty list when no input data."""
        span = create_llm_span()
        span["input"] = {}
        messages = await extract_input_messages(span, Provider.OPENAI)

        assert messages == []


class TestExtractOutput:
    """Tests for extract_output function."""

    @pytest.mark.asyncio
    async def test_extract_output_messages(self) -> None:
        """Extract output from meta.output.messages."""
        span = create_llm_span(
            output_messages=[{"role": "assistant", "content": "The answer is 42."}]
        )
        output = await extract_output(span)

        assert output is not None
        assert output.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_extract_output_with_tool_calls(self) -> None:
        """Extract output with tool calls."""
        span = create_llm_span(
            output_messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "SF"}',
                            },
                        }
                    ],
                }
            ]
        )
        output = await extract_output(span)

        assert output is not None
        assert output.choices[0].stop_reason == "tool_calls"
        msg = output.choices[0].message
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function == "get_weather"

    @pytest.mark.asyncio
    async def test_extract_output_value_fallback(self) -> None:
        """Fall back to output.value."""
        span = create_llm_span()
        span["output"] = {"value": "Simple response"}
        output = await extract_output(span)

        assert output is not None


class TestExtractUsage:
    """Tests for extract_usage function."""

    def test_extract_usage(self) -> None:
        """Extract token usage from metrics."""
        span = create_llm_span(input_tokens=100, output_tokens=50, total_tokens=150)
        usage = extract_usage(span)

        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_no_metrics_returns_none(self) -> None:
        """Return None when no token metrics."""
        span = create_llm_span()
        span["metrics"] = {}
        usage = extract_usage(span)

        assert usage is None

    def test_partial_metrics(self) -> None:
        """Handle partial metrics (only input_tokens)."""
        span = create_llm_span()
        span["metrics"] = {"input_tokens": 50}
        usage = extract_usage(span)

        assert usage is not None
        assert usage.input_tokens == 50
        assert usage.output_tokens == 0


class TestExtractTools:
    """Tests for extract_tools function."""

    def test_extract_tool_definitions(self) -> None:
        """Extract tool definitions from metadata."""
        span = create_llm_span(
            metadata={
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather for a city",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {
                                        "type": "string",
                                        "description": "City name",
                                    }
                                },
                                "required": ["city"],
                            },
                        },
                    }
                ]
            }
        )
        tools = extract_tools(span)

        assert len(tools) == 1
        assert tools[0].name == "get_weather"

    def test_no_tools_returns_empty(self) -> None:
        """Return empty list when no tools in metadata."""
        span = create_llm_span()
        tools = extract_tools(span)

        assert tools == []


class TestSumTokens:
    """Tests for sum_tokens function."""

    def test_sum_across_spans(self) -> None:
        """Sum tokens across multiple spans."""
        span1 = create_llm_span(input_tokens=10, output_tokens=5)
        span2 = create_llm_span(input_tokens=20, output_tokens=10)

        total = sum_tokens([span1, span2])
        assert total == 45

    def test_prefers_total_tokens(self) -> None:
        """Use total_tokens when available instead of input + output."""
        span = create_llm_span(input_tokens=10, output_tokens=5, total_tokens=20)
        assert sum_tokens([span]) == 20

    def test_empty_spans(self) -> None:
        """Sum of empty list is 0."""
        assert sum_tokens([]) == 0


class TestNormalizeMessages:
    """Tests for _normalize_messages falsy-args handling."""

    @pytest.mark.parametrize(
        "args_value",
        ["", {}],
        ids=["empty-string", "empty-dict"],
    )
    def test_falsy_args_falls_back_to_arguments(self, args_value: object) -> None:
        """Falsy args (empty string or empty dict) fall back to arguments."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "fn",
                        "args": args_value,
                        "arguments": '{"x": 1}',
                    }
                ],
            }
        ]
        result = _normalize_messages(messages)
        tc = result[0]["tool_calls"][0]
        assert tc["function"]["arguments"] == '{"x": 1}'

    def test_falsy_args_without_arguments_defaults_to_empty(self) -> None:
        """Falsy args without arguments key defaults to '{}'."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"name": "fn", "args": ""}],
            }
        ]
        result = _normalize_messages(messages)
        tc = result[0]["tool_calls"][0]
        assert tc["function"]["arguments"] == "{}"

    def test_absent_args_falls_back_to_arguments(self) -> None:
        """Missing args key falls back to arguments."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "fn",
                        "arguments": '{"x": 1}',
                    }
                ],
            }
        ]
        result = _normalize_messages(messages)
        tc = result[0]["tool_calls"][0]
        assert tc["function"]["arguments"] == '{"x": 1}'

    @pytest.mark.parametrize(
        ("args_value", "expected"),
        [
            ([1, 2, 3], "[1, 2, 3]"),
            (42, "42"),
            (0, "0"),
            # Note: bool is a subtype of int in Python (isinstance(True, int)
            # is True), so branch ordering in _normalize_messages matters.
            (True, "true"),
            (False, "false"),
        ],
        ids=["list", "int", "zero", "bool-true", "bool-false"],
    )
    def test_non_string_non_dict_args_serialized(
        self, args_value: object, expected: str
    ) -> None:
        """Non-string, non-dict args are JSON-serialized."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"name": "fn", "args": args_value}],
            }
        ]
        result = _normalize_messages(messages)
        tc = result[0]["tool_calls"][0]
        assert tc["function"]["arguments"] == expected

    def test_stale_arguments_key_removed(self) -> None:
        """Stale arguments key should not remain on the tool call dict."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "fn",
                        "args": "",
                        "arguments": '{"x": 1}',
                    }
                ],
            }
        ]
        result = _normalize_messages(messages)
        tc = result[0]["tool_calls"][0]
        assert "arguments" not in tc, "stale 'arguments' key should be removed"
        assert "arguments" in tc["function"]


class TestSimpleMessageConversion:
    """Tests for _simple_message_conversion fallback."""

    def test_converts_all_roles(self) -> None:
        """Convert system, user, and assistant messages."""
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = _simple_message_conversion(messages)

        assert len(result) == 3
        assert result[0].role == "system"
        assert result[1].role == "user"
        assert result[2].role == "assistant"

    def test_skips_unknown_roles(self) -> None:
        """Messages with unrecognized roles are skipped."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "tool", "content": "result"},
        ]
        result = _simple_message_conversion(messages)

        assert len(result) == 1
        assert result[0].role == "user"

    def test_empty_messages(self) -> None:
        """Empty input produces empty output."""
        assert _simple_message_conversion([]) == []

    def test_missing_content_defaults_to_empty(self) -> None:
        """Missing content field defaults to empty string."""
        messages = [{"role": "user"}]
        result = _simple_message_conversion(messages)

        assert len(result) == 1
        assert result[0].content == ""


class TestParseToolSchema:
    """Tests for _parse_tool_schema with various input formats."""

    def test_string_json_input(self) -> None:
        """Parse a JSON string tool schema."""
        import json

        schema = json.dumps(
            {
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                }
            }
        )
        result = _parse_tool_schema(schema)

        assert result is not None
        assert result.name == "search"
        assert result.description == "Search the web"

    def test_invalid_json_string_returns_none(self) -> None:
        """Invalid JSON string returns None."""
        assert _parse_tool_schema("not json") is None

    def test_non_dict_non_str_returns_none(self) -> None:
        """Non-dict, non-string input returns None."""
        assert _parse_tool_schema(42) is None
        assert _parse_tool_schema(None) is None


class TestExtractToolCalls:
    """Tests for _extract_tool_calls with edge cases."""

    def test_non_dict_entries_skipped(self) -> None:
        """Non-dict entries in tool_calls list are skipped."""
        data = [
            "not a dict",
            42,
            {
                "id": "call_1",
                "function": {"name": "fn", "arguments": "{}"},
            },
        ]
        result = _extract_tool_calls(data)  # type: ignore[arg-type]

        assert len(result) == 1
        assert result[0].function == "fn"

    def test_non_dict_function_skipped(self) -> None:
        """Tool call with non-dict function field is skipped."""
        data = [{"id": "call_1", "function": "not_a_dict"}]
        result = _extract_tool_calls(data)

        assert len(result) == 0


class TestExtractInputMessagesGoogle:
    """Tests for Google provider message conversion path."""

    @pytest.mark.asyncio
    async def test_extract_google_messages(self) -> None:
        """Google provider normalizes 'model' role and routes through OpenAI converter."""
        span = create_llm_span(
            model_provider="google",
            model_name="gemini-1.5-pro",
            input_messages=[
                {"role": "user", "content": "Hello"},
                {"role": "model", "content": "Hi there"},
            ],
        )
        messages = await extract_input_messages(span, Provider.GOOGLE)

        assert len(messages) >= 1
        assert any(m.role == "user" for m in messages)
        assert any(m.role == "assistant" for m in messages)


class TestExtractSystemText:
    """Tests for _extract_system_text with different content formats."""

    def test_string_content(self) -> None:
        """Extract system text from string content."""
        messages = [{"role": "system", "content": "Be helpful."}]
        assert _extract_system_text(messages) == "Be helpful."

    def test_list_of_text_blocks(self) -> None:
        """Extract system text from list of text block dicts."""
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Be helpful."},
                    {"type": "text", "text": "Be concise."},
                ],
            }
        ]
        assert _extract_system_text(messages) == "Be helpful. Be concise."

    def test_list_of_plain_strings(self) -> None:
        """Extract system text from list of plain strings."""
        messages = [
            {
                "role": "system",
                "content": ["Be helpful.", "Be concise."],
            }
        ]
        assert _extract_system_text(messages) == "Be helpful. Be concise."

    def test_mixed_list_content(self) -> None:
        """Extract system text from mixed list of blocks and strings."""
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "First."},
                    "Second.",
                ],
            }
        ]
        assert _extract_system_text(messages) == "First. Second."

    def test_no_system_message(self) -> None:
        """Return None when no system message exists."""
        messages = [{"role": "user", "content": "Hello"}]
        assert _extract_system_text(messages) is None

    def test_empty_list_content(self) -> None:
        """Return None for empty list content."""
        messages = [{"role": "system", "content": []}]
        assert _extract_system_text(messages) is None


class TestRestructureAnthropicContent:
    """Tests for restructure_anthropic_content content parser."""

    @pytest.mark.parametrize(
        ("description", "input_content", "expected_types"),
        [
            (
                "tool_result with text",
                '[tool_result (tool_use_id: toolu_abc)]'
                '[{"type":"text","text":"result here"}]'
                "[/tool_result]",
                [("tool_result", "toolu_abc")],
            ),
            (
                "tool_result with image",
                '[tool_result (tool_use_id: toolu_img)]'
                '[{"type":"text","text":"desc"},'
                '{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBOR"}}]'
                "[/tool_result]",
                [("tool_result", "toolu_img")],
            ),
            (
                "multiple tool results",
                '[tool_result (tool_use_id: toolu_1)]'
                '[{"type":"text","text":"first"}]'
                "[/tool_result]"
                " "
                '[tool_result (tool_use_id: toolu_2)]'
                '[{"type":"text","text":"second"}]'
                "[/tool_result]",
                [("tool_result", "toolu_1"), ("tool_result", "toolu_2")],
            ),
        ],
        ids=["text-result", "image-result", "multiple-results"],
    )
    def test_tool_result_parsing(
        self,
        description: str,
        input_content: str,
        expected_types: list[tuple[str, str]],
    ) -> None:
        """Parse tool_result wrappers into structured blocks."""
        messages = [{"role": "user", "content": input_content}]
        result = restructure_anthropic_content(messages)

        content = result[0]["content"]
        assert isinstance(content, list)
        tool_results = [b for b in content if b["type"] == "tool_result"]
        assert len(tool_results) == len(expected_types)
        for block, (expected_type, expected_id) in zip(
            tool_results, expected_types, strict=True
        ):
            assert block["type"] == expected_type
            assert block["tool_use_id"] == expected_id
            assert isinstance(block["content"], list)

    def test_tool_result_with_image_has_image_block(self) -> None:
        """Image data inside tool_result is preserved as an image block."""
        content = (
            '[tool_result (tool_use_id: toolu_img)]'
            '[{"type":"text","text":"screenshot"},'
            '{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBOR"}}]'
            "[/tool_result]"
        )
        messages = [{"role": "user", "content": content}]
        result = restructure_anthropic_content(messages)

        inner = result[0]["content"][0]["content"]
        types = [b["type"] for b in inner]
        assert "image" in types

    def test_raw_json_list(self) -> None:
        """Parse raw JSON list of typed blocks."""
        blocks = [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
        messages = [{"role": "user", "content": json.dumps(blocks)}]
        result = restructure_anthropic_content(messages)

        assert result[0]["content"] == blocks

    def test_plain_text_unchanged(self) -> None:
        """Plain text content passes through unchanged."""
        messages = [{"role": "user", "content": "just plain text"}]
        result = restructure_anthropic_content(messages)

        assert result[0]["content"] == "just plain text"

    def test_invalid_json_unchanged(self) -> None:
        """Invalid JSON content passes through unchanged."""
        messages = [{"role": "user", "content": "{not valid json"}]
        result = restructure_anthropic_content(messages)

        assert result[0]["content"] == "{not valid json"

    def test_non_string_content_unchanged(self) -> None:
        """Already-structured content passes through unchanged."""
        structured = [{"type": "text", "text": "already structured"}]
        messages = [{"role": "user", "content": structured}]
        result = restructure_anthropic_content(messages)

        assert result[0]["content"] is structured

    def test_text_between_tool_results_captured(self) -> None:
        """Text between tool_result blocks is captured as text blocks."""
        content = (
            "preamble "
            '[tool_result (tool_use_id: toolu_1)][{"type":"text","text":"r1"}][/tool_result]'
            " middle "
            '[tool_result (tool_use_id: toolu_2)][{"type":"text","text":"r2"}][/tool_result]'
            " epilogue"
        )
        messages = [{"role": "user", "content": content}]
        result = restructure_anthropic_content(messages)

        blocks = result[0]["content"]
        text_blocks = [b for b in blocks if b["type"] == "text"]
        text_values = [b["text"] for b in text_blocks]
        assert "preamble" in text_values
        assert "middle" in text_values
        assert "epilogue" in text_values


class TestAnthropicInputMessages:
    """End-to-end tests for Anthropic provider input message extraction."""

    @pytest.mark.asyncio
    async def test_stringified_tool_result_produces_tool_message(self) -> None:
        """Stringified tool_result in user message produces ChatMessageTool."""
        from inspect_ai.model._chat_message import ChatMessageTool

        span = create_llm_span(
            model_provider="anthropic",
            model_name="claude-sonnet-4-20250514",
            input_messages=[
                {"role": "user", "content": "What's in this image?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check."},
                        {
                            "type": "tool_use",
                            "id": "toolu_abc",
                            "name": "screenshot",
                            "input": {},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": (
                        '[tool_result (tool_use_id: toolu_abc)]'
                        '[{"type":"text","text":"Screenshot taken"}]'
                        "[/tool_result]"
                    ),
                },
            ],
        )
        messages = await extract_input_messages(span, Provider.ANTHROPIC)

        tool_msgs = [m for m in messages if isinstance(m, ChatMessageTool)]
        assert len(tool_msgs) >= 1
        assert tool_msgs[0].tool_call_id == "toolu_abc"

    @pytest.mark.asyncio
    async def test_stringified_image_in_tool_result_produces_content_image(self) -> None:
        """Base64 image inside stringified tool_result produces ContentImage."""
        from inspect_ai.model import ContentImage

        image_data = "iVBORw0KGgoAAAANSUhEUg=="
        inner = json.dumps(
            [
                {"type": "text", "text": "screenshot"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data,
                    },
                },
            ]
        )
        span = create_llm_span(
            model_provider="anthropic",
            model_name="claude-sonnet-4-20250514",
            input_messages=[
                {"role": "user", "content": "Describe this."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_img",
                            "name": "capture",
                            "input": {},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": (
                        f"[tool_result (tool_use_id: toolu_img)]{inner}[/tool_result]"
                    ),
                },
            ],
        )
        messages = await extract_input_messages(span, Provider.ANTHROPIC)

        # Find content blocks across all messages.
        all_content: list[object] = []
        for m in messages:
            if isinstance(m.content, list):
                all_content.extend(m.content)

        image_blocks = [c for c in all_content if isinstance(c, ContentImage)]
        assert len(image_blocks) >= 1


class TestAnthropicOutputExtraction:
    """End-to-end tests for Anthropic provider output extraction."""

    @pytest.mark.asyncio
    async def test_structured_tool_use_produces_tool_calls(self) -> None:
        """Structured Anthropic output with tool_use produces ToolCall objects."""
        span = create_llm_span(
            model_provider="anthropic",
            model_name="claude-sonnet-4-20250514",
            output_messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me search for that."},
                        {
                            "type": "tool_use",
                            "id": "toolu_search",
                            "name": "web_search",
                            "input": {"query": "weather SF"},
                        },
                    ],
                }
            ],
        )
        output = await extract_output(span, Provider.ANTHROPIC)

        msg = output.choices[0].message
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) >= 1
        assert msg.tool_calls[0].function == "web_search"
        assert msg.tool_calls[0].arguments == {"query": "weather SF"}

    @pytest.mark.asyncio
    async def test_text_only_anthropic_output(self) -> None:
        """Text-only Anthropic output works through the Anthropic path."""
        span = create_llm_span(
            model_provider="anthropic",
            model_name="claude-sonnet-4-20250514",
            output_messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "The answer is 42."},
                    ],
                }
            ],
        )
        output = await extract_output(span, Provider.ANTHROPIC)

        msg = output.choices[0].message
        assert "42" in str(msg.content)

    @pytest.mark.asyncio
    async def test_string_content_anthropic_output_falls_through(self) -> None:
        """String content on Anthropic output still produces valid ModelOutput."""
        span = create_llm_span(
            model_provider="anthropic",
            model_name="claude-sonnet-4-20250514",
            output_messages=[
                {"role": "assistant", "content": "Simple text response"},
            ],
        )
        output = await extract_output(span, Provider.ANTHROPIC)

        assert output is not None
        assert output.model == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_backward_compat_no_provider(self) -> None:
        """extract_output without provider still works (backward compat)."""
        span = create_llm_span(
            output_messages=[{"role": "assistant", "content": "Hello!"}]
        )
        output = await extract_output(span)

        assert output is not None
        assert output.model == "gpt-4o"
