import asyncio
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import subprocess

import chainlit as cl

import messages


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

    async def confirm_and_run(self, tool_call_id: str, arguments: dict):
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

        return messages.ToolOutputMessage(tool_call_id, self.name, response, canceled)


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

        return out


TOOLS = [BashTool(), PythonTool()]
