#!/usr/bin/env python3

"""A bash assistant that can run commands and answer questions about the system."""

import base64
import difflib
import enum
import io
import os
from pprint import pprint
import re
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from random import choice
from typing import Annotated, Literal


import openai
import requests
import rich
import typer
import yaml
from rich import print as rprint

import constants
from utils import (
    ai_query,
    ai_stream,
    fmt_diff,
    get_text_input,
    confirm_action,
)
import commits
from ynab import app as ynab_app
from dcron import dcron
from ovh_tools import app as ovh_app


def run_suggested_command(command: str, bash_console) -> tuple[str, str]:
    """Run a command suggested by the assistant. Return the possibly edited command and the output."""
    try:
        to_run = bash_console.prompt(default=command.strip())
    except KeyboardInterrupt:
        return command, "Command was cancelled by the user."

    # Add the command to the user's zsh history.
    zsh_history_path = os.environ.get("HISTFILE", "~/.zsh_history")
    zsh_history_path = Path(zsh_history_path).expanduser()
    # Format the command to be added to the history.
    with open(zsh_history_path, "a") as f:
        f.write(f": {int(time.time())}:0;{to_run.replace("\n", "\\\n")}\n")

    # Run the command while streaming the output to the terminal and capturing it.
    with style("response"):
        try:
            print(f"Running command: {to_run=!r}")
            with subprocess.Popen(
                to_run.strip(), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            ) as proc:
                output = ""
                for line in proc.stdout:
                    line = line.decode()
                    print(line, end="")
                    output += line
        except KeyboardInterrupt:
            output += "^C Interrupted"

    return to_run, output


def get_response(
    system: str | None,
    messages: list[dict],
    end_after: str = constants.BASH_END,
    model: str = constants.OPENAI_MODEL,
) -> str:
    with style("assistant"):
        answer = ""
        for text in ai_stream(system, messages, model=model):
            answer += text
            print(text, end="", flush=True)
            if end_after in answer:
                answer = answer[: answer.find(end_after)] + end_after
                break
    print()

    return answer


@contextmanager
def style(kind: Literal["system", "assistant", "response"]):
    match kind:
        case "system":
            print(constants.STYLE_SYSTEM + "System: ", end="")
        case "assistant":
            print(constants.STYLE_ASSISTANT + "Assistant: ", end="")
        case "response":
            print(constants.STYLE_RESPONSE, end="")
        case _:
            raise ValueError(f"Unknown style: {kind}")

    yield

    print(constants.STYLE_END, end="")


# --------------------- CLI --------------------- #


app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def callback(ctx: typer.Context, anthropic: bool = False):
    """Interaction with openai's and anthropic's API."""

    constants.USE_OPENAI = not anthropic

    # Check that the API key is set.
    # os.environ.setdefault("OPENAI_API_KEY", "")  # You can add it here too
    if constants.USE_OPENAI and not os.environ.get("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environement variable to your API key: 'sk-...'.")
        print("Or add it in the code, two lines above this message.")
        exit(1)

    # If no command is given, run the bash assistant.
    if ctx.invoked_subcommand is None:
        bash_scaffold()


@app.command(name="bash")
def bash_scaffold(model: str = "claude-sonnet-4-20250514"):
    """A bash assistant that can run commands and answer questions about the system."""

    import prompt_toolkit as pt
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.lexers import PygmentsLexer
    from pygments.lexers.shell import BashLexer

    BASH_CONSOLE = pt.PromptSession(
        message="run: ",
        history=FileHistory(constants.CACHE_DIR / "commands.txt"),
        auto_suggest=AutoSuggestFromHistory(),
        vi_mode=constants.VI_MODE,
        lexer=PygmentsLexer(BashLexer),
    )

    PROMPTS_CONSOLE = pt.PromptSession(
        message="> ",
        history=FileHistory(constants.CACHE_DIR / "history.txt"),
        auto_suggest=AutoSuggestFromHistory(),
        vi_mode=constants.VI_MODE,
    )

    # Load pre-defined prompts.
    with open(constants.SRC / "prompts.yaml") as f:
        PROMPTS = yaml.safe_load(f)

    from InquirerPy import inquirer

    prompt = inquirer.fuzzy(
        message="Select a prompt",
        choices=[
            *PROMPTS,
        ],
    ).execute()
    system = PROMPTS.get(prompt, prompt)

    if system:
        with style("system"):
            print(system)
    else:
        system = None

    messages = []

    last_command_result = ""
    while True:
        question = PROMPTS_CONSOLE.prompt()
        messages.append({"role": "user", "content": last_command_result + question})

        answer = get_response(system, messages, model=model)

        # Ask to run the bash command
        if (start := answer.find(constants.BASH_START)) != -1 and (
            end := answer.find(constants.BASH_END, start)
        ) != -1:

            start += len(constants.BASH_START)
            command = answer[start:end]

            command, output = run_suggested_command(command, BASH_CONSOLE)

            # Replace the command with the one that was actually run.
            answer = answer[:start] + command + answer[end:]
            # edited = typer.edit(output) if edited is not None:
            #     output = edited
            last_command_result = f"\n<response>{output}</response>"
        else:
            last_command_result = ""

        # We do it at the end, because the user might have edited the command.
        messages.append({"role": "assistant", "content": answer})


@app.command(name="prompts")
def list_prompts():
    """List the available prompts."""

    with open(constants.PROMPTS_FILE) as f:
        prompts = yaml.safe_load(f)

    for name, prompt in prompts.items():
        print(f"{name}:")
        print(prompt)
        print()


@app.command(name="translate")
def translate(text: str, to: str = "french"):
    """Translate the given text to the given language."""

    system = f"Give a short definition in {to} of the given term without using a translation of the term, then suggest 3-6 {to} translations."
    response = ai_query(system, text)
    print(response)


@app.command(name="fix")
def fix_typos(
    text: Annotated[str, typer.Argument()] = None,
    show_diff: bool = True,
    color: bool = True,
    heavy: bool = False,
):
    """Fix typos in the given text."""

    text = get_text_input(text)

    system = """
You are given a text and you need to fix the language (typos, grammar, ...).
If needed, fix also the formatting and ensure the text is gender neutral. HEAVY
Output directly the corrected text, without any comment.
""".strip()

    if heavy:
        system = system.replace(
            "HEAVY",
            "Please reformulate the text when needed, use better words and make it more clear when possible.",
        )
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
def generate_image(
    prompt: Annotated[str, typer.Argument()] = None,
    hd: bool = False,
    vivid: bool = True,
    output: Path = None,
    horizontal: bool = False,
    vertical: bool = False,
    show: bool = False,
):
    """Generate an image from the given prompt."""

    assert not (horizontal and vertical), "Cannot be both horizontal and vertical."

    prompt = get_text_input(prompt)

    print("Generating image...")
    response = openai.images.generate(
        prompt=prompt,
        model="dall-e-3",
        n=1,
        response_format="b64_json",
        quality="hd" if hd else "standard",
        style="vivid" if vivid else "natural",
        size="1792x1024" if horizontal else "1024x1792" if vertical else "1024x1024",
    )

    b64 = response.data[0].b64_json
    revised_prompt = response.data[0].revised_prompt

    print("Initial prompt:")
    print(prompt)

    print("Revised prompt:")
    print(revised_prompt)

    # Save the image with the extension from output.
    import PIL.Image

    img = PIL.Image.open(io.BytesIO(base64.b64decode(b64)))

    if output is None:
        date = time.strftime("%Y-%m-%d_%H-%M-%S")
        title = re.sub(r"[^a-zA-Z0-9]+", "_", revised_prompt[:60])
        file = f"{date}_{title}.png"
        output = Path("~/Pictures/dalle3").expanduser() / file
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving image to {str(output)}")

    img.save(output)

    add_exif(output, prompt, revised_prompt)

    if show:
        subprocess.run(["imv", output])


@app.command(hidden=True)
def add_exif(img_path: Path, prompt: str, revised_prompt: str):
    """Add the given prompts to the image's Exif metadata."""

    import PIL.Image
    import piexif.helper

    img = PIL.Image.open(img_path)
    if "exif" in img.info:
        exif_dict = piexif.load(img.info["exif"])
    else:
        exif_dict = {"0th": {}, "Exif": {}}
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(prompt)
    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = revised_prompt
    exif_bytes = piexif.dump(exif_dict)
    img.save(img_path, exif=exif_bytes)


@app.command(hidden=True)
def show_exif(png_path: Path):
    """Show the exif metadata of the given png image."""

    import PIL.Image
    from rich import print
    import piexif.helper

    img = PIL.Image.open(png_path)
    exif_dict = piexif.load(img.info["exif"])

    user_comment_raw = exif_dict["Exif"].get(piexif.ExifIFD.UserComment, b"")
    user_comment = piexif.helper.UserComment.load(user_comment_raw)

    image_description = exif_dict["0th"].get(piexif.ImageIFD.ImageDescription, b"").decode("utf-8")

    print("Original prompt:", user_comment)
    print("Revised prompt: ", image_description)


VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer", "random"]
Voice = enum.StrEnum("Voices", {v: v for v in VOICES})


@app.command(name="speak")
def speak(
    text: Annotated[str, typer.Argument()] = None,
    voice: Voice = Voice.random,
    output: str = None,
    speed: float = 1.0,
    quiet: bool = False,
):
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
        model="tts-1-hd",
        voice=voice,
        response_format="mp3",
        speed=speed,
    )
    response.stream_to_file(output)
    # Play the audio.
    if not quiet:
        subprocess.run(["cvlc", "--play-and-exit", "--no-volume-save", "--no-loop", output])


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
        },
    )

    if response.status_code == 200 or response.status_code == 202:
        print("🎉 Email reported as spam.")
    else:
        print(f"❌ Error: {response.status_code} {response.reason}")
        rich.print(response.json())


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def web(ctx: typer.Context, new: bool = False):
    """Start a web server to interact with the assistant."""

    if new:
        command = "chainlit run main.py --watch".split()
        os.chdir(constants.ROOT / "chaty")
    else:
        command = "streamlit run src/web.py --server.runOnSave True".split()
        # Make sure the command is run in the correct directory.
        os.chdir(constants.ROOT)

    # Find the correct python executable.
    python = sys.executable
    full_command = [python, "-m", *command, *ctx.args]

    os.execvp(full_command[0], full_command)


@app.command()
def timer(
    duration: list[str],
    bell: Path = Path("~/Media/bell.mp3").expanduser(),
    ring_count: int = -1,
    ring_interval: int = 10,
):
    """Start a timer for the given duration, e.g. '5m' or '1h 10s'."""

    # Parse the duration.
    total = 0
    multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    for part in duration:
        try:
            total += int(part[:-1]) * multiplier[part[-1]]
        except (ValueError, KeyError):
            print(f"Invalid duration part: {part}")
            quit(1)

    # Show a visual timer with Textual.
    from textual.app import App
    from textual.widgets import Footer, Static
    from textual.reactive import reactive
    import pyfiglet

    fonts_names = [
        # Largest to smallest
        "univers",
        "colossal",
        "alphabet",
        "4max",
        "3x5",
    ]
    fonts = [pyfiglet.Figlet(font=font) for font in fonts_names]

    def fmt_duration(sec: int) -> str:
        if sec < 0:
            sign = "-"
            sec = -sec
        else:
            sign = ""

        minutes, sec = divmod(sec, 60)
        hours, minutes = divmod(minutes, 60)

        if hours:
            return f"{sign}{hours:02.0f}:{minutes:02.0f}:{sec:02.0f}"
        else:
            return f"{sign}{minutes:02.0f}:{sec:02.0f}"

    class TimeDisplay(Static):
        start_time = reactive(time.time)
        time_shown = reactive(0)

        def __init__(self):
            self.duration = total
            self.last_rung = 0
            self.ring_count = 0
            super().__init__()

        def on_mount(self):
            self.set_interval(1 / 24, self.update_time_shown)
            # self.styles.text_align = "center"  # No. it strips the text before centering
            self.styles.content_align_vertical = "middle"
            self.styles.height = "100h"

        def update_time_shown(self):
            time_since_start = time.time() - self.start_time
            sec = self.duration - time_since_start
            self.time_shown = int(sec)

        def watch_time_shown(self):
            if self.time_shown < 0:
                if self.ring_count != ring_count and time.time() - self.last_rung > ring_interval:
                    self.last_rung = time.time()
                    self.ring_count += 1
                    subprocess.Popen(["paplay", bell])
            elif self.time_shown > 0:
                self.last_rung = 0
                self.ring_count = 0

            sec_str = fmt_duration(self.time_shown)
            # What space do we have?
            width = self.size.width
            height = self.size.height - 2

            # Find the largest font that fits.
            bw, _bh = len(sec_str), 1
            best = [sec_str]
            log = ""
            for font in fonts:
                new = font.renderText(sec_str)
                lines = new.split("\n")
                h = len(lines)
                w = max(len(line) for line in lines)
                log += f"{w=} {h=} {font.font}\n"
                if bw <= w <= width and h <= height:
                    best = lines
                    bw, _bh = w, h

            first_line = "\n".join(line.center(width) for line in best)
            time_since_start = time.time() - self.start_time
            second_line = f"Total: {fmt_duration(time_since_start)}".center(width)

            self.update(first_line + "\n\n" + second_line)

    class TimerApp(App):
        BINDINGS = [
            ("j", "sub_minute", "Sub 1 min"),
            ("k", "add_minute", "Add 1 min"),
        ]

        def __init__(self, duration: int):
            super().__init__()

        def compose(self):
            yield TimeDisplay()
            yield Footer()

        def add_time(self, amount: int):
            time_display = self.query_one(TimeDisplay)
            time_display.duration += amount

        def action_add_minute(self):
            self.add_time(60)

        def action_sub_minute(self):
            self.add_time(-60)

    TimerApp(total).run()


def sanitize_file_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "'- ." else "" for c in name)


@app.command(name="logseq")
def to_logseq(
    url: str,
    logseq_dir: Path = Path("~/Documents/logseq").expanduser(),
    template: Path = Path("~/Documents/logseq/pages/Paper Template.md").expanduser(),
):
    """Download the given paper and convert it to a Logseq journal entry."""

    # See docs at https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/get_graph_get_paper
    # In particular, availaible fields
    id = f"URL:{url}"
    api_url = f"https://api.semanticscholar.org/graph/v1/paper/{id}"
    fields = "title,citationStyles,authors,abstract,year,journal,openAccessPdf,publicationDate,venue,citationCount"

    print("Using Semantic Scholar API to fetch the paper data...")
    response = requests.get(api_url, params={"fields": fields})
    if response.status_code != 200:
        print(response.text)
        raise Exception(f"Failed to fetch data: {response.status_code}")
    data = response.json()

    data["citations"] = len(data.get("citations", []))
    pprint(data)

    # Gather all the fields for templating
    data = {
        "url": url,
        "title": data["title"],
        "authors": [author["name"] for author in data["authors"]],
        "abstract": data["abstract"],
        "year": data["year"],
        # "pdf": f"https://arxiv.org/pdf/{data['arxivId']}.pdf",
        "pdf_url": data["openAccessPdf"]["url"] if data["openAccessPdf"] else None,
        "logseq_author_listing": ", ".join(f"[[{author['name']}]]" for author in data["authors"]),
        "publication_date": data["publicationDate"],
        "journal": data["journal"]["name"] if "journal" in data else "<no journal>",
        "venue": data["venue"],
        "bibtex": data["citationStyles"]["bibtex"],
        "citations": data["citationCount"],
        "today": time.strftime("[[%Y-%m-%d, %a]]"),
    }
    for i, author in enumerate(data["authors"], start=1):
        data[f"author_{i}"] = author

    file_format = "{title} - {author_1}"
    file_name = sanitize_file_name(file_format.format(**data))

    pdf_relative_path = Path("assets") / "Papers" / f"{file_name}.pdf"
    md_relative_path = Path("pages") / "Papers" / f"{file_name}.md"
    pdf_path = logseq_dir / pdf_relative_path
    entry_path = logseq_dir / md_relative_path

    data["filename_without_extension"] = file_name
    data["pdf_relative_path"] = pdf_relative_path
    data["md_relative_path"] = md_relative_path
    data["hls_filename"] = "hls__" + file_name

    # Create the Logseq entry. One page, with metadata and the PDF.
    template_str = template.read_text()
    entry = template_str.format(**data)

    pprint(data)

    print("📝 Entry:")
    print(f"\033[33m{entry}\033[0m")

    pdf_url = data["pdf_url"]
    if pdf_url is None:
        rprint("[red]Could not find a pdf automatically![/]")
        pdf_url = input("Please provide an url or a local path to the PDF: ")

    rprint(f"🖊 Writting to [yellow]{pdf_path}[/] and [yellow]{entry_path}[/]")
    if entry_path.exists():
        print("⚠ File {entry_path} already exists. Overwrite?")
    if pdf_path.exists():
        print("⚠ File {pdf_path} already exists. Overwrite?")

    if not confirm_action("Do you want to continue?"):
        exit(1)

    if pdf_url.startswith("http"):
        print("📄 Downloading PDF...")
        response = requests.get(data["pdf_url"])
        pdf_path.write_bytes(response.content)
    else:  # Just copy
        pdf_path.write_bytes(Path(pdf_url).read_bytes())

    entry_path.write_text(entry)

    print(f"✅ Saved as [[{file_name}]] in Logseq.")


app.add_typer(ynab_app, name="ynab", no_args_is_help=True, help="Commands to facilitate YNAB.")
app.add_typer(
    dcron.app, name="dcron", no_args_is_help=True, help="My own cron jobs. Stuff that repeats."
)
app.add_typer(ovh_app, name="ovh", no_args_is_help=True, help="Commands to interact OVH DNS.")
app.command()(commits.commit)
app.command()(commits.commit_install)

if __name__ == "__main__":
    app()
