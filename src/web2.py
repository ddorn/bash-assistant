import asyncio
import base64
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
import json
from pprint import pprint
import subprocess
import warnings
import PIL.Image
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message import ChatCompletionMessage

import chainlit as cl

# cl.instrument_openai()

client = AsyncOpenAI()


class Tool:
    name: str
    description: str
    parameters: dict
    required: list

    def as_json(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
            "strict": True,
        }

    def confirm_msg(self, **kwargs):
        return f"Confirm run of `{self.name}` with arguments: {kwargs}"

    async def run(self, **kwargs) -> str:
        raise NotImplementedError()

    async def confirm_and_run(self, tool_call_id: str, arguments: dict, message_history: list):
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
            tool_output = response = await self.run(**arguments)
            await cl.Message(content=response, author=self.name).send()
        else:
            response = "Tool call denied by user."
            tool_output = None

        message_history.append(
            {
                "role": "tool",
                "name": self.name,
                "content": response,
                "tool_call_id": tool_call_id,
            }
        )

        return tool_output


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
        stdout, _ = await output.communicate()
        return stdout.decode(errors="replace")


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
        try:
            with redirect_stdout(StringIO()) as stdout, redirect_stderr(StringIO()) as stderr:
                exec(code)
                output = stdout.getvalue()
                error = stderr.getvalue()
        except Exception as e:
            return f"Error {e} while running code.\nStdout:\n{output}\n\nStderr:\n{error}"
        return output


TOOLS = [BashTool(), PythonTool()]


@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )


@cl.step(type="tool")
async def call_tool(tool: Tool, tool_call_id: str, arguments: dict, message_history: list):

    current_step = cl.context.current_step
    assert current_step is not None
    current_step.name = tool.name
    current_step.input = arguments

    function_response = await tool.run(**arguments)

    message_history.append(
        {
            "role": "tool",
            "name": tool.name,
            "content": function_response,
            "tool_call_id": tool_call_id,
        }
    )

    return function_response


@cl.step(type="llm")
async def call_gpt(messages: list) -> ChatCompletionMessage:
    cl.context.current_step.name = "GPT-4o"
    cl.context.current_step.input = messages

    answer = (
        (
            await client.chat.completions.create(
                messages=messages,
                model="gpt-4o",
                temperature=0.2,
                tools=[tool.as_json() for tool in TOOLS],
            )
        )
        .choices[0]
        .message
    )

    messages.append(answer.model_dump())

    return answer


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    assert isinstance(message_history, list)

    parts = []
    for element in message.elements:
        if element.mime.startswith("image"):
            img = PIL.Image.open(element.path)
            # Convert the image to PNG format
            buffered = BytesIO()
            img.save(buffered, format="PNG")

            # Encode the image to base64
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            url = "data:image/png;base64," + img_base64
            parts.append(dict(type="image_url", image_url=dict(url=url)))
        else:
            warnings.warn(f"Unsupported element type: {element.mime}")

    if message.content:
        parts.append(message.content)

    message_history.append({"role": "user", "content": parts})
    answer = await call_gpt(message_history)

    if answer.content:
        await cl.Message(content=answer.content, author="Answer").send()

    if answer.tool_calls:
        for tool in answer.tool_calls:
            # Confirm the function call
            the_tool = next(t for t in TOOLS if t.name == tool.function.name)
            args = json.loads(tool.function.arguments)
            tool_call_id = tool.id

            response = await the_tool.confirm_and_run(tool_call_id, args, message_history)
            if response is None:
                continue
            else:
                pass

    pprint(message_history)
