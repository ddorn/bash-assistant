import asyncio
import base64
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import dataclasses
from io import BytesIO, StringIO
import json
from pprint import pprint
import subprocess
import PIL.Image
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from anthropic import AsyncAnthropic
import itertools

import chainlit as cl
from chainlit.input_widget import Select

import chainlit.data

# cl.instrument_openai()

client = AsyncOpenAI()
client_antropic = AsyncAnthropic()


class Tool:
    name: str
    description: str
    parameters: dict
    required: list

    def as_json(self, for_anthropic: bool = False):
        shema = {
            "type": "object",
            "properties": self.parameters,
            "required": self.required,
        }
        if for_anthropic:
            return {
                "name": self.name,
                "description": self.description,
                "input_schema": shema,
            }
        else:
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": shema,
                },
                "strict": True,
            }

    def confirm_msg(self, **kwargs):
        return f"Confirm run of `{self.name}` with arguments: {kwargs}"

    async def run(self, **kwargs) -> str:
        raise NotImplementedError()

    async def confirm_and_run(self, tool_call_id: str, arguments: dict) -> "ToolOutputMessage":
        await cl.Message(content=self.confirm_msg(**arguments), author=self.name).send()
        confirmation = await cl.AskActionMessage(
            content="Confirm?",
            actions=[
                cl.Action(name="continue", value="continue", label="✅"),
                cl.Action(name="cancel", value="cancel", label="❌"),
            ],
            timeout=1_000_000,
        ).send()

        if confirmation and confirmation.get("value") == "continue":
            response = await self.run(**arguments)
            canceled = False
        else:
            response = "Tool call denied by user."
            canceled = True

        return ToolOutputMessage(tool_call_id, self.name, response, canceled)


class BashTool(Tool):
    name = "run_bash_command"
    description = "You are being run in a scaffold on an archlinux machine running bash. You can access any file and program on the machine. Some of the tools available to you are: convert, pdftk, swaymsg, signal-cli, jq, and more."
    parameters = {
        "command": {
            "type": "string",
            "description": "The command to run in the bash shell",
        }
    }
    required = ["command"]

    def confirm_msg(self, command: str):
        return f"Running the following command? \n```bash\n{command}\n```"

    async def run(self, command: str):
        output = await asyncio.create_subprocess_shell(
            command.strip(), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        stdout, stderr = await output.communicate()
        out = stdout.decode(errors="replace")
        return f"```\n{out}\n```"


class PythonTool(Tool):
    name = "run_python_code"
    description = "Run any python code and return its stdout.\nYou have access to the standard library plus scipy, numpy, pandas, plotly, and more. Install packages only if importing them fails.\n\nExample: `import numpy as np; print(np.random.rand(5))`"
    parameters = {
        "code": {
            "type": "string",
            "description": "The python code to run",
        }
    }
    required = ["code"]

    def confirm_msg(self, code: str):
        return f"Running the following python code? \n```python\n{code}\n```"

    async def run(self, code: str):
        with redirect_stdout(StringIO()) as stdout, redirect_stderr(StringIO()) as stderr:
            try:
                exec(code)
            except Exception:
                pass

            output = stdout.getvalue()
            error = stderr.getvalue()

        out = ""
        if output:
            out += f"stdout:\n```\n{output}\n```"
        if error:
            out += f"stderr:\n```\n{error}\n```"
        if not out:
            out = "No output from the tool. Remember to use print() to output something."


TOOLS = [BashTool(), PythonTool()]


@dataclass
class MessagePart:
    pass

    def to_openai(self):
        raise NotImplementedError()

    def to_anthropic(self):
        raise NotImplementedError()


@dataclass
class TextMessage(MessagePart):
    text: str
    is_user: bool

    def to_openai(self):
        return {
            "role": "user" if self.is_user else "assistant",
            "content": [{"type": "text", "text": self.text}],
        }

    to_anthropic = to_openai


@dataclass
class ImageMessage(MessagePart):
    image_url: str

    def to_openai(self):
        return {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": self.image_url}}],
        }

    to_anthropic = to_openai


@dataclass
class ToolRequestMessage(MessagePart):
    name: str
    parameters: dict
    id: str

    def to_openai(self):
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": self.id,
                    "type": "function",
                    "function": {"name": self.name, "arguments": self.parameters},
                }
            ],
        }

    def to_anthropic(self):
        return {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "name": self.name,
                    "input": self.parameters,
                    "id": self.id,
                }
            ],
        }


@dataclass
class ToolOutputMessage(MessagePart):
    id: str
    name: str
    content: str
    canceled: bool = False

    def to_openai(self):
        return {
            "role": "tool",
            "name": self.name,
            "content": self.content,
            "tool_call_id": self.id,
        }

    def to_anthropic(self):
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": self.id,
                    "content": self.content,
                }
            ],
        }


def merge[T: (dict, list)](a: T, b: T) -> T:
    if isinstance(a, list) and isinstance(b, list):
        return a + b
    elif isinstance(a, dict) and isinstance(b, dict):
        new = {}
        for key in a.keys():
            new[key] = a[key]
        for key in b.keys():
            if key in new and isinstance(new[key], (dict, list)):
                new[key] = merge(new[key], b[key])
            elif key in new:
                assert new[key] == b[key]
            else:
                new[key] = b[key]
        return new
    else:
        raise ValueError(f"Cannot merge {a} and {b}")


class MessageHistory(list[MessagePart]):

    def to_openai(self) -> list[ChatCompletionMessage]:
        formated = []
        # For openai, we need to merge:
        # - an optional assistant TextMessage and the consecutive ToolRequestMessages into a single one
        # - a user TextMessage and subsequent ImageMessages from the same user into a single one

        i = 0
        while i < len(self):
            message = self[i]

            # Merge consecutive user text message and Image messages
            if isinstance(message, TextMessage) and message.is_user:
                new = message.to_openai()
                i += 1
                for attached_image in itertools.takewhile(
                    lambda x: isinstance(x, ImageMessage), self[i:]
                ):
                    new = merge(new, attached_image.to_openai())
                    i += 1

            # Merge an assistant message with subsequent tool requests
            elif isinstance(message, TextMessage) and not message.is_user:
                new = message.to_openai()
                i += 1
                for tool_request in itertools.takewhile(
                    lambda x: isinstance(x, ToolRequestMessage), self[i:]
                ):
                    new = merge(new, tool_request.to_openai())
                    i += 1

            # Just add the message
            else:
                new = message.to_openai()
                i += 1

            formated.append(new)

        return formated

    def to_anthropic(self) -> list[dict]:
        formated = []

        # For anthropic, we need to merge:
        # - all user messages (text and image) and tool outputs into a single message
        # - all other, ie: all assistant messages and tool requests into a single message

        def is_user_message(x):
            return (
                isinstance(x, TextMessage)
                and x.is_user
                or isinstance(x, ImageMessage)
                or isinstance(x, ToolOutputMessage)
            )

        i = 0
        while i < len(self):
            message = self[i]

            # Merge consecutive user text/image message and tool outputs
            if is_user_message(message):
                new = message.to_anthropic()
                i += 1
                for part in itertools.takewhile(is_user_message, self[i:]):
                    new = merge(new, part.to_anthropic())
                    i += 1
            else:
                new = message.to_anthropic()
                i += 1
                for part in itertools.takewhile(lambda x: not is_user_message(x), self[i:]):
                    new = merge(new, part.to_anthropic())
                    i += 1

            formated.append(new)

        return formated

    def to_simple_json(self) -> list[dict]:
        return [dataclasses.asdict(message) for message in self]


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
                    messages=messages.to_openai(),
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
            img = PIL.Image.open(element.path)
            # Convert the image to PNG format
            buffered = BytesIO()
            img.save(buffered, format="PNG")

            # Encode the image to base64
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            url = "data:image/png;base64," + img_base64
            message_history.append(ImageMessage(url))
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
