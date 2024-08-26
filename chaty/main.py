from pprint import pprint
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

import chainlit as cl

from messages import (
    ImageMessage,
    MessageHistory,
    MessagePart,
    TextMessage,
    ToolRequestMessage,
)
from tools import Tool, TOOLS
from llms import OpenAILLM, AnthropicLLM

client = AsyncOpenAI()
client_antropic = AsyncAnthropic()


SYSTEM = """
You are an helpful assistant, tasked to help Diego in his daily tasks. Your answer are concise and you use tools only when necessary.
You can use code blocks and any markdown formatting.
""".strip()


MODELS = [
    AnthropicLLM("Claude 3.5 Sonnet", "claude-3-5-sonnet-20240620"),
    OpenAILLM("GPT 4o", "gpt-4o"),
    OpenAILLM("GPT 4o mini", "gpt-4o-mini"),
]


@cl.on_chat_start
async def start_chat():
    cl.user_session.set("message_history", MessageHistory())
    cl.user_session.set("tool_use", True)
    cl.user_session.set("model", MODELS[0].nice_name)

    settings, actions = make_settings_widget()
    settings_message = cl.Message(settings, actions=actions)
    await settings_message.send()
    cl.user_session.set("settings_message", settings_message)


def make_settings_widget():
    next_model = get_next_model()
    tool_use = cl.user_session.get("tool_use", True)

    actions = [
        cl.Action(name="switch-model", value=next_model, label=f"Switch to {next_model}"),
        cl.Action(name="toggle-tool-use", value="", label="Toggle tool use"),
    ]

    settings = f"""
- **Model**: {cl.user_session.get("model")}
- **Tool Use**: {'❌✅'[tool_use]}
""".strip()

    return settings, actions


async def update_settings_widget():
    settings, actions = make_settings_widget()
    settings_message: cl.Message = cl.user_session.get("settings_message")
    settings_message.content = settings
    for action in settings_message.actions:
        await action.remove()
    settings_message.actions = actions
    await settings_message.update()


@cl.action_callback("switch-model")
async def switch_model_action(action):
    model = action.value
    cl.user_session.set("model", model)

    await update_settings_widget()


@cl.action_callback("toggle-tool-use")
async def toggle_tool_use_action(action):
    tool_use = not cl.user_session.get("tool_use", True)
    cl.user_session.set("tool_use", tool_use)

    await update_settings_widget()


def get_next_model() -> str:
    current = cl.user_session.get("model", MODELS[0].nice_name)
    model_idx = next(i for i, m in enumerate(MODELS) if m.nice_name == current)
    next_model = MODELS[(model_idx + 1) % len(MODELS)]
    return next_model.nice_name


async def call_gpt(messages: MessageHistory) -> list[MessagePart]:
    nice_model_name = cl.user_session.get("model")
    model = next(m for m in MODELS if m.nice_name == nice_model_name)

    new_messages: list[MessagePart] = []

    with cl.Step(name=f"Calling {model}", type="llm"):

        new_messages = await model(SYSTEM, messages, TOOLS)

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
