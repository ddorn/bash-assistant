#!/usr/bin/env python3

"""A bash assistant that can run commands and answer questions about the system."""

from contextlib import contextmanager
import os
import subprocess
import time
from typing import Literal

import openai
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.shell import BashLexer
from prompt_toolkit.key_binding import KeyBindings
import prompt_toolkit as pt
from pathlib import Path

# Check that the API key is set.
os.environ.setdefault("OPENAI_API_KEY", "")  # You can add it here too
if not (key := os.environ.get("OPENAI_API_KEY")):
    print("Please set the OPENAI_API_KEY environement variable to your API key: 'sk-...'.")
    print("Or add it in the code, two lines above this message.")
    exit(1)

MODEL = "gpt-4-1106-preview"

VI_MODE = True
BASH_START = "<bash>"
BASH_END = "</bash>"

STYLE_SYSTEM = "\033[38;5;250m"
STYLE_ASSISTANT = "\033[38;5;39m"
STYLE_RESPONSE = "\033[38;5;208m"
STYLE_END = "\033[0m"

SYSTEM_PROMPT = f"""
You are being run in a scaffold on an archlinux machine running bash. You can access any file and program on the machine.
When you need to run a bash command, wrap it in {BASH_START} and {BASH_END} tags.
You will be shown the output of the command, but you cannot interact with it.
Your answers are concise. You don't ask the user before running a command, just run it. You assume most things that the user did not specify. When you need more info, you use cat, ls or pwd.
Don't provide explanations unless asked.
""".strip()

CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "bash-assistant"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASH_CONSOLE = pt.PromptSession(
    message="run: ",
    history=FileHistory(CACHE_DIR / "commands.txt"),
    auto_suggest=AutoSuggestFromHistory(),
    vi_mode=VI_MODE,
    lexer=PygmentsLexer(BashLexer),
)

BINDINGS = KeyBindings()
PROMPTS_CONSOLE = pt.PromptSession(
    message="> ",
    history=FileHistory(CACHE_DIR / "history.txt"),
    auto_suggest=AutoSuggestFromHistory(),
    vi_mode=VI_MODE,
    key_bindings=BINDINGS,
)


RECORDING = None

@BINDINGS.add("c-a")
def get_audio_input(event):
    """Get audio input from the user and add it to the prompt.

    The first call starts recording, the second call stops it and adds the transcript to the prompt.
    """
    global RECORDING
    msg = "🎤 Recording... press Ctrl+A to stop"

    if RECORDING is None:
        # Show that we are recording.
        event.current_buffer.insert_text(msg)
        # Run in parallel to avoid blocking the main thread.

        proc = subprocess.Popen(
            "ffmpeg -f alsa -i default -acodec libmp3lame -ab 128k -y temp.mp3",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        RECORDING = proc
        return

    RECORDING.terminate()
    RECORDING.wait()
    RECORDING = None

    transcript = openai.audio.transcriptions.create(
        model="whisper-1",
        file=open("temp.mp3", "rb")
    )
    os.remove("temp.mp3")

    # Remove the recording message.
    event.current_buffer.delete_before_cursor(len(msg))
    # Add the transcript to the prompt.
    event.current_buffer.insert_text(transcript.text)


def run_suggested_command(command: str) -> tuple[str, str]:
    """Run a command suggested by the assistant. Return the possibly edited command and the output."""
    try:
        to_run = BASH_CONSOLE.prompt(default=command)
    except KeyboardInterrupt:
        return command, "Command was cancelled by the user."

    # Add the command to the user's zsh history.
    zsh_history_path = os.environ.get("HISTFILE", "~/.zsh_history")
    zsh_history_path = Path(zsh_history_path).expanduser()
    # Format the command to be added to the history.
    to_run = to_run.replace("\n", "\\\n")
    with open(zsh_history_path, "a") as f:
        f.write(f": {int(time.time())}:0;{to_run}\n")


    # Run the command while streaming the output to the terminal and capturing it.
    with style("response"):
        try:
            with subprocess.Popen(to_run, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
                output = ""
                for line in proc.stdout:
                    line = line.decode()
                    print(line, end="")
                    output += line
        except KeyboardInterrupt:
            output += "^C Interrupted"

    return to_run, output


def get_response(messages: list[dict], end_after: str = BASH_END) -> str:
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
    )

    with style("assistant"):
        answer = ""
        for chunk in response:
            text = chunk.choices[0].delta.content
            if text is None:
                break

            answer += text
            print(text, end="", flush=True)
            if end_after in answer:
                answer = answer[:answer.find(end_after)] + end_after
                break
    print()

    return answer


@contextmanager
def style(kind: Literal["system", "assistant", "response"]):
    match kind:
        case "system":
            print(STYLE_SYSTEM + "System: ", end="")
        case "assistant":
            print(STYLE_ASSISTANT + "Assistant: ", end="")
        case "response":
            print(STYLE_RESPONSE, end="")
        case _:
            raise ValueError(f"Unknown style: {kind}")

    yield

    print(STYLE_END, end="")


def main():

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    last_command_result = ""

    with style("system"):
        print(SYSTEM_PROMPT)

    while True:
        question = PROMPTS_CONSOLE.prompt()
        messages.append({
            "role": "user",
            "content": last_command_result + question
        })

        answer = get_response(messages)

        # Ask to run the bash command
        if (start := answer.find(BASH_START)) != -1 and \
                (end := answer.find(BASH_END, start)) != -1:

            start += len(BASH_START)
            command = answer[start:end]

            command, output = run_suggested_command(command)

            # Replace the command with the one that was actually run.
            answer = answer[:start] + command + answer[end:]
            last_command_result = f"\n<response>{output}</response>"
        else:
            last_command_result = ""

        # We do it at the end, because the user might have edited the command.
        messages.append({"role": "assistant", "content": answer})


if __name__ == '__main__':
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        pass
