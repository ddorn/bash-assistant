import json
import anthropic
import openai
from tools import Tool
from messages import MessageHistory, MessagePart, TextMessage, ToolRequestMessage


class LLM:
    def __init__(
        self,
        nice_name: str,
        model_name: str,
    ):
        self.nice_name = nice_name
        self.model_name = model_name

    async def __call__(
        self, system: str, messages: MessageHistory, tools: list[Tool]
    ) -> list[MessagePart]:
        raise NotImplementedError()


class OpenAILLM(LLM):
    def __init__(self, nice_name: str, model_name: str):
        super().__init__(nice_name, model_name)
        self.client = openai.AsyncOpenAI()

    async def __call__(
        self, system: str, messages: MessageHistory, tools: list[Tool]
    ) -> list[MessagePart]:
        answer = (
            (
                await self.client.chat.completions.create(
                    messages=[
                        dict(role="system", content=system),
                        *messages.to_openai(),
                    ],  # type: ignore
                    model=self.model_name,
                    temperature=0.2,
                    tools=[tool.as_json() for tool in tools],
                )
            )
            .choices[0]
            .message
        )

        new_messages = []
        if answer.content is not None:
            if isinstance(answer.content, str):
                new_messages.append(TextMessage(answer.content, is_user=False))
            else:
                new_messages.append(
                    TextMessage(f"Unrecognized content type: {answer.content}", is_user=False)
                )

        if answer.tool_calls:
            for tool_call in answer.tool_calls:
                new_messages.append(
                    ToolRequestMessage(
                        tool_call.function.name,
                        json.loads(tool_call.function.arguments),
                        tool_call.id,
                    )
                )

        return new_messages


class AnthropicLLM(LLM):
    def __init__(self, nice_name: str, model_name: str):
        super().__init__(nice_name, model_name)
        self.client = anthropic.AsyncAnthropic()

    async def __call__(
        self, system: str, messages: MessageHistory, tools: list[Tool]
    ) -> list[MessagePart]:

        answer = await self.client.messages.create(
            system=system,
            messages=messages.to_anthropic(),
            model=self.model_name,
            temperature=0.2,
            max_tokens=4096,
            tools=[tool.as_json(for_anthropic=True) for tool in tools],
        )

        new_messages = []
        for part in answer.content:
            if part.type == "text":
                new_messages.append(TextMessage(part.text, is_user=False))
            elif part.type == "tool_use":
                new_messages.append(ToolRequestMessage(part.name, part.input, part.id))
            else:
                new_messages.append(
                    TextMessage(f"Unrecognized content type: {part.type}", is_user=False)
                )

        return new_messages
