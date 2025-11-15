    def _parse_response(self, response: anthropic.types.Message) -> LLMResponse:
        """Parse Anthropic response into LLMResponse.

        Args:
            response: Anthropic Message response

        Returns:
            LLMResponse object
        """
        # Extract text content, thinking, and tool calls
        text_content = ""
        thinking_content = ""
        tool_calls = []

        # Handle both string and list content types
        if isinstance(response.content, str):
            # If content is a string, use it directly
            text_content = response.content
        else:
            # If content is a list of blocks, parse each block
            for block in response.content:
                if block.type == "text":
                    text_content += block.text
                elif block.type == "thinking":
                    thinking_content += block.thinking
                elif block.type == "tool_use":
                    # Parse Anthropic tool_use block
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            type="function",
                            function=FunctionCall(
                                name=block.name,
                                arguments=block.input,
                            ),
                        )
                    )

        return LLMResponse(
            content=text_content,
            thinking=thinking_content if thinking_content else None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=response.stop_reason or "stop",
        )
