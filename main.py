#!/usr/bin/env python3
# %%

"""A bash assistant that can run commands and answer questions about the system."""

from contextlib import contextmanager
import difflib
import base64
import io
import os
from random import choice
import re
import subprocess
import time
from typing import Literal
import click
import tempfile

import PIL.Image
import openai
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.shell import BashLexer
from prompt_toolkit.key_binding import KeyBindings
import prompt_toolkit as pt
from pathlib import Path

from constants import *
from utils import ai_query, fmt_diff, get_text_input

# Check that the API key is set.
os.environ.setdefault("OPENAI_API_KEY", "")  # You can add it here too
if not (key := os.environ.get("OPENAI_API_KEY")):
    print("Please set the OPENAI_API_KEY environement variable to your API key: 'sk-...'.")
    print("Or add it in the code, two lines above this message.")
    exit(1)



SYSTEM_PROMPT = f"""
You are being run in a scaffold on an archlinux machine running bash. You can access any file and program on the machine.
When you need to run a bash command, wrap it in {BASH_START} and {BASH_END} tags.
You will be shown the output of the command, but you cannot interact with it.
Your answers are concise. You don't ask the user before running a command, just run it. You assume most things that the user did not specify. When you need more info, you use cat, ls or pwd.
Don't provide explanations unless asked.
""".strip()


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


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    # If no command is given, run the bash assistant.
    if ctx.invoked_subcommand is None:
        ctx.invoke(bash_scaffold)


@cli.command(name="no")
@click.pass_context
def no(ctx):
    """Run the bash assistant without the prompt."""
    ctx.invoke(bash_scaffold, no_prompt=True)


@cli.command(name="bash")
@click.option("-n", "--no-prompt", is_flag=True, help="Don't use a prompt.")
def bash_scaffold(no_prompt: bool):

    if no_prompt:
        messages = []
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        with style("system"):
            print(SYSTEM_PROMPT)

    last_command_result = ""
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


@cli.command(name="fix")
@click.argument("text", type=str, default=None, required=False)
@click.option("-c", "--clean-only", is_flag=True, help="Only print the cleaned text")
@click.option("-n", "--no-color", is_flag=True, help="Don't colorize the diff. Implies --clean-only")
def fix_typos(text: str, clean_only: bool, no_color: bool):
    """Fix typos in the given text."""

    text = get_text_input(text)

    system = """
You are given a text and you need to fix the language (typos, grammar, ...).
If needed, fix also the formatting.
Output directly the corrected text, without any comment.
""".strip()

    corrected = ai_query(system, text)

    if no_color:
        print(corrected)
        return

    # Compute the difference between the two texts
    words1 = re.findall(r"(\w+|\W+)", text.strip())
    words2 = re.findall(r"(\w+|\W+)", corrected.strip())

    diff = difflib.ndiff(words1, words2)
    past, new = fmt_diff(diff)

    if not clean_only:
        print(past)
        print()
    print(new)


@cli.command(name="img")
@click.argument("prompt", type=str, default=None, required=False)
@click.option("--hd", is_flag=True, help="Generate an HD image.")
@click.option("--vivid/--natural", default=True, help="Generate a vivid or natural image.")
@click.option("-o", "--output", type=click.Path(dir_okay=False, writable=True))
@click.option("-s", "--show", is_flag=True, help="Show the generated image.")
def generate_image(prompt: str, hd: bool, vivid: bool, output: str | None, show: bool):
    """Generate an image from the given prompt."""

    # %%
    prompt = "a ghost made of colorful circles"
    hd = False
    vivid = True
    output = "out.png"
    show = True
    # %%
    prompt = get_text_input(prompt)

    print("Generating image...")
    response = openai.images.generate(
        prompt=prompt,
        model="dall-e-3",
        n=1,
        response_format="b64_json",
        quality='hd' if hd else "standard",
        style='vivid' if vivid else "natural",
    )

    if output is None:
        # Use a temporary file.
        output = tempfile.mktemp(suffix=".png", prefix="openai-img-")

    # %%

    b64 = response.data[0].b64_json
    revised_prompt = response.data[0].revised_prompt

    print("Initial prompt:")
    print(prompt)

    print("Revised prompt:")
    print(revised_prompt)

    # Save the image with the extension from output.
    img = PIL.Image.open(io.BytesIO(base64.b64decode(b64)))
    img.save(output)

    if show:
        subprocess.run(["imv", output])

# %%


VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

@cli.command(name="speak")
@click.argument("text", type=str, default=None, required=False)
@click.option("--voice", type=click.Choice(VOICES))
@click.option("-o", "--output", type=click.Path(dir_okay=False, writable=True))
@click.option("-s", "--speed", type=float, default=1.0)
@click.option("-q", "--quiet", is_flag=True, help="Don't speak the generated audio.")
def speak(text: str, voice: str | None, output: str | None, speed: float, quiet: bool):
    """Speak the given text."""

    text = get_text_input(text)

    if voice is None:
        voice = choice(VOICES)

    if output is None:
        # Use a temporary file.
        output = tempfile.mktemp(suffix=".mp3", prefix="openai-voice-")

    print(f"Generating audio... {voice=}, {speed=}, {output=}")
    response = openai.audio.speech.create(
        input=text,
        model='tts-1-hd',
        voice=voice,
        response_format='mp3',
        speed=speed,
    )
    response.stream_to_file(output)
    # Play the audio.
    if not quiet:
        subprocess.run(["cvlc", "--play-and-exit", "--no-volume-save", "--no-loop", output])


if __name__ == '__main__':
    cli()
