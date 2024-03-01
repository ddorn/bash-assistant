#!/usr/bin/env python3

"""A bash assistant that can run commands and answer questions about the system."""

import difflib
import base64
import enum
import io
import os
import re
import subprocess
import sys
import time
import tempfile
from contextlib import contextmanager
from typing import Annotated, Literal
from random import choice
from pathlib import Path
import rich

import typer
import openai
import pandas as pd
from config import SIGNAL_SPAM_USER

from constants import *
from utils import ai_query, fmt_diff, get_text_input



def run_suggested_command(command: str, bash_console) -> tuple[str, str]:
    """Run a command suggested by the assistant. Return the possibly edited command and the output."""
    try:
        to_run = bash_console.prompt(default=command)
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


RECORDING = None


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



# --------------------- CLI --------------------- #

app = typer.Typer()

@app.callback(invoke_without_command=True)
def callback(ctx: typer.Context):
    """Interaction with openai's API."""


    # Check that the API key is set.
    # os.environ.setdefault("OPENAI_API_KEY", "")  # You can add it here too
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environement variable to your API key: 'sk-...'.")
        print("Or add it in the code, two lines above this message.")
        exit(1)



    # If no command is given, run the bash assistant.
    if ctx.invoked_subcommand is None:
        bash_scaffold(no_prompt=True)


@app.command(name="bash")
def bash_scaffold(no_prompt: bool = False):
    """A bash assistant that can run commands and answer questions about the system."""

    import prompt_toolkit as pt
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.lexers import PygmentsLexer
    from pygments.lexers.shell import BashLexer
    from prompt_toolkit.key_binding import KeyBindings

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
    BINDINGS.add("c-a")(get_audio_input)
    PROMPTS_CONSOLE = pt.PromptSession(
        message="> ",
        history=FileHistory(CACHE_DIR / "history.txt"),
        auto_suggest=AutoSuggestFromHistory(),
        vi_mode=VI_MODE,
        key_bindings=BINDINGS,
    )


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

            command, output = run_suggested_command(command, BASH_CONSOLE)

            # Replace the command with the one that was actually run.
            answer = answer[:start] + command + answer[end:]
            last_command_result = f"\n<response>{output}</response>"
        else:
            last_command_result = ""

        # We do it at the end, because the user might have edited the command.
        messages.append({"role": "assistant", "content": answer})


@app.command(name="translate")
def translate(text: str, to: str = "french"):
    """Translate the given text to the given language."""

    system = f"Give a short definition in {to} of the given term without using a translation of the term, then suggest 3-6 {to} translations."
    response = ai_query(system, text)
    print(response)



@app.command(name="fix")
def fix_typos(text: Annotated[str, typer.Argument()] = None,
              show_diff: bool = True,
              color: bool = True,
              heavy: bool = False):
    """Fix typos in the given text."""

    text = get_text_input(text)

    system = """
You are given a text and you need to fix the language (typos, grammar, ...).
If needed, fix also the formatting and ensure the text is gender neutral. HEAVY
Output directly the corrected text, without any comment.
""".strip()

    if heavy:
        system = system.replace("HEAVY", "Please reformulate the text when needed, use better words and make it more clear when possible.")
    else:
        system = system.replace(" HEAVY", "")

    corrected = ai_query(system, text)

    if not color:
        print(corrected)
        return

    # Compute the difference between the two texts
    words1 = re.findall(r"(\w+|\W+)", text.strip())
    words2 = re.findall(r"(\w+|\W+)", corrected.strip())

    diff = difflib.ndiff(words1, words2)
    past, new = fmt_diff(diff)

    if show_diff:
        print(past)
        print()
    print(new)


@app.command(name="img")
def generate_image(prompt: Annotated[str, typer.Argument()] = None,
                   hd: bool = False,
                   vivid: bool = True,
                   output: Path = None,
                   show: bool = False):
    """Generate an image from the given prompt."""

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
        output = Path(tempfile.mktemp(suffix=".png", prefix="openai-img-"))
        print(f"Saving image to {str(output)}...")

    b64 = response.data[0].b64_json
    revised_prompt = response.data[0].revised_prompt

    print("Initial prompt:")
    print(prompt)

    print("Revised prompt:")
    print(revised_prompt)

    # Save the image with the extension from output.
    import PIL.Image
    from PIL.ExifTags import Base
    img = PIL.Image.open(io.BytesIO(base64.b64decode(b64)))

    img.save(output)
    # add_exif(output, prompt, revised_prompt)

    if show:
        subprocess.run(["imv", output])



VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer", 'random']
Voice = enum.StrEnum("Voices", {v: v for v in VOICES})


@app.command(name="speak")
def speak(text: Annotated[str, typer.Argument()] = None,
          voice: Voice = Voice.random,
          output: str = None,
          speed: float = 1.0,
          quiet: bool = False):
    """Speak the given text."""

    text = get_text_input(text)

    if voice is Voice.random:
        voice = choice([v for v in Voice if v is not Voice.random])

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




@app.command(hidden=True)
def add_exif(img_path: Path, prompt: str, revised_prompt: str):
    """Add the given prompts to the image's Exif metadata."""

    show_exif(img_path)

    import PIL.Image
    from PIL.ExifTags import Base

    img = PIL.Image.open(img_path)

    # Save the prompts in the image's Exif metadata, using the fields "UserComment" and "ImageDescription".
    img._exif[Base.UserComment] = prompt.encode()
    img._exif[Base.ImageDescription] = revised_prompt.encode()
    img.save(img_path, exif=img._exif)

    show_exif(img_path)


@app.command(hidden=True)
def show_exif(png_path: Path):
    """Show the exif metadata of the given png image."""

    import PIL.Image
    from rich import print

    img = PIL.Image.open(png_path)
    exif = img.getexif()
    print(exif)
    print(img.info)


@app.command()
def bas2ynab(input_file: Path, output_file: Path):
    """Convert a bank record from the BAS to the CSV format of YNAB."""

    df = pd.read_csv(input_file.open(encoding="latin1"),
                     skiprows=11,
                      sep=";", dtype=str)

    assert list(df.columns) == ["Date", "Libellé", "Montant", "Valeur"]

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%y").dt.strftime("%m/%d/%Y")
    out["Payee"] = df["Libellé"]
    out["Memo"] = ""
    out["Amount"] = df["Montant"]

    print(out)
    out.to_csv(output_file, index=False)

    print(f"🎉 Converted {input_file} to {output_file}")


@app.command()
def report_spam():
    """Report the piped email as spam using signal-spam."""

    import requests
    import config

    email = sys.stdin.buffer.read()
    assert email, "No email provided."

    response = requests.post(
        url="https://www.signal-spam.fr/api/signaler",
        timeout=120,
        auth=(config.SIGNAL_SPAM_USER, config.SIGNAL_SPAM_PASS),
        data={
            "message": base64.b64encode(email),
        }
    )

    if response.status_code == 200 or response.status_code == 202:
        print("🎉 Email reported as spam.")
    else:
        print(f"❌ Error: {response.status_code} {response.reason}")
        rich.print(response.json())


if __name__ == '__main__':
    app()