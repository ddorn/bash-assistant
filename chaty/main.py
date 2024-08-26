from pprint import pprint

import chainlit as cl

from messages import (
    ImageMessage,
    MessageHistory,
    MessagePart,
    TextMessage,
    ToolRequestMessage,
)
from tools import Tool, TOOLS
from settings import Settings


SYSTEM = """
You are an helpful assistant, tasked to help Diego in his daily tasks. Your answer are concise and you use tools only when necessary.
You can use code blocks and any markdown formatting.
""".strip()


@cl.on_chat_start
async def start_chat():
    await Settings.get().update_settings_message()


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    return cl.User(
        identifier=username or "Diego",
        display_name=username or "Diego",
    )


async def call_gpt(messages: MessageHistory) -> list[MessagePart]:
    model = Settings.get().current_model

    with cl.Step(name=model.nice_name, type="llm") as step:
        step.input = messages.to_dict()

        new_messages = await model(SYSTEM, messages, TOOLS)

        step.output = MessageHistory(new_messages).to_dict()

        return new_messages


@cl.on_message
async def on_message(message: cl.Message):
    settings = Settings.get()
    message_history = settings.history

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

    pprint(message_history.to_dict())


if __name__ == "__main__":
    messages = [
        TextMessage("yo", True),
        ImageMessage("https://example.com"),
        TextMessage("assis", False),
        TextMessage("assis", False),
        TextMessage("https://example.com", True),
    ]

    pprint(MessageHistory(messages).to_openai())
