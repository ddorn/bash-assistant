import json
from pprint import pprint
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

import chainlit as cl
from chainlit.input_widget import Select

from messages import (
    ImageMessage,
    MessageHistory,
    MessagePart,
    TextMessage,
    ToolRequestMessage,
    ToolOutputMessage,
)
from tools import Tool, TOOLS

client = AsyncOpenAI()
client_antropic = AsyncAnthropic()


SYSTEM = """
You are an helpful assistant, tasked to help Diego in his daily tasks. Your answer are concise and you use tools only when necessary.
You can use code blocks and any markdown formatting.
""".strip()


@cl.on_chat_start
async def start_chat():
    cl.user_session.set("message_history", MessageHistory())

    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Model",
                values=[
                    "claude-3-5-sonnet-20240620",
                    "gpt-4o",
                    "gpt-4o-mini",
                ],
                initial_index=0,
            ),
        ]
    ).send()

    cl.user_session.set("settings", settings)
    await cl.Message(content=f"Selected model: {settings}").send()


@cl.on_settings_update
async def update_settings(settings: dict):
    cl.user_session.set("settings", settings)
    await cl.Message(content=f"Updated settings: {settings}").send()


@cl.step(type="tool")
async def call_tool(
    tool: Tool, tool_call_id: str, arguments: dict, message_history: MessageHistory
):

    current_step = cl.context.current_step
    assert current_step is not None
    current_step.name = tool.name
    current_step.input = arguments

    function_response = await tool.run(**arguments)

    message_history.append(ToolOutputMessage(tool_call_id, tool.name, function_response))

    return function_response


@cl.step(type="llm")
async def call_gpt(messages: MessageHistory) -> list[MessagePart]:
    model = cl.user_session.get("settings")["model"]
    cl.context.current_step.name = model

    new_messages: list[MessagePart] = []

    cl.context.current_step.input = messages.to_simple_json()
    if "gpt" in model:
        answer = (
            (
                await client.chat.completions.create(
                    messages=[
                        dict(role="system", content=SYSTEM),
                        *messages.to_openai(),
                    ],  # type: ignore
                    model=model,
                    temperature=0.2,
                    tools=[tool.as_json() for tool in TOOLS],
                )
            )
            .choices[0]
            .message
        )

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

    elif "claude" in model:

        answer = await client_antropic.messages.create(
            system=SYSTEM,
            messages=messages.to_anthropic(),
            model=model,
            temperature=0.2,
            max_tokens=4096,
            tools=[tool.as_json(for_anthropic=True) for tool in TOOLS],
        )

        for part in answer.content:
            if part.type == "text":
                new_messages.append(TextMessage(part.text, is_user=False))
            elif part.type == "tool_use":
                new_messages.append(ToolRequestMessage(part.name, part.input, part.id))
            else:
                new_messages.append(
                    TextMessage(f"Unrecognized content type: {part.type}", is_user=False)
                )

    cl.context.current_step.output = MessageHistory(new_messages).to_simple_json()

    return new_messages


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    assert isinstance(message_history, MessageHistory)

    for element in message.elements:
        if element.mime.startswith("image"):
            message_history.append(ImageMessage.from_path(element.path))
        else:
            await cl.Message(f"Unsupported element type: {element.mime}").send()

    if message.content:
        message_history.append(TextMessage(message.content, is_user=True))

    new_message_parts = await call_gpt(message_history)
    message_history.extend(new_message_parts)

    for part in new_message_parts:
        if isinstance(part, ToolRequestMessage):
            # Confirm the function call
            the_tool: Tool = next(t for t in TOOLS if t.name == part.name)

            output = await the_tool.confirm_and_run(part.id, part.parameters)
            if not output.canceled:
                await cl.Message(content=output.content, author=the_tool.name).send()
            message_history.append(output)

        elif isinstance(part, TextMessage):
            await cl.Message(content=part.text).send()

        else:
            await cl.Message(content=f"Unrecognized message part: {part}").send()

    pprint(message_history.to_simple_json())


if __name__ == "__main__":
    messages = [
        TextMessage("yo", True),
        ImageMessage("https://example.com"),
        TextMessage("assis", False),
        TextMessage("assis", False),
        TextMessage("https://example.com", True),
    ]

    pprint(MessageHistory(messages).to_openai())
