import os
import subprocess

from openai import OpenAI
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.shell import BashLexer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import message_dialog
import prompt_toolkit as pt
from pathlib import Path

MODEL = "gpt-4"

VI_MODE = True
BASH_START = "<bash>"
BASH_END = "</bash>"

STYLE_SYSTEM = "\033[38;5;240m"
STYLE_ASSISTANT = "\033[38;5;39m"
STYLE_RESPONSE = "\033[38;5;208m"
STYLE_END = "\033[0m"

SYSTEM = f"""
You are being run in a scaffold on an archlinux machine running bash. You can access any file and program on the machine.
When you need to run a bash command, wrap it in {BASH_START} and {BASH_END} tags.
You will be shown the output of the command, but you cannot interact with it.
Your answers are concise. You don't ask the user before running a command, just run it. Don't provide explanations unless asked.
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

    client = OpenAI()

    transcript = client.audio.transcriptions.create(
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

    # Run the command while streaming the output to the terminal and capturing it.
    print(STYLE_RESPONSE, end="")
    try:
        with subprocess.Popen(to_run, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
            output = ""
            for line in proc.stdout:
                line = line.decode()
                print(line, end="")
                output += line
    except KeyboardInterrupt:
        output += "^C Interrupted"

    print(STYLE_END, end="")

    return to_run, output


def stream_response(response):
    answer = ""
    print(STYLE_ASSISTANT + "Assistant: ", end="")
    for chunk in response:
        text = chunk.choices[0].delta.content
        if text is None:
            break

        answer += text
        print(text, end="", flush=True)
        if BASH_END in answer:
            answer = answer.rstrip()
            break
    print(STYLE_END)
    return answer


def main():
    client = OpenAI()

    messages = [{"role": "system", "content": SYSTEM}]
    executed = ""

    print(f"{STYLE_SYSTEM}System: {SYSTEM}{STYLE_END}")
    while True:
        question = executed + PROMPTS_CONSOLE.prompt()
        messages.append({"role": "user", "content": question})

        answer = stream_response(client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
        ))

        # Ask to run the bash command
        if (start := answer.find(BASH_START)) != -1 and \
                (end := answer.find(BASH_END, start)) != -1:

            start += len(BASH_START)
            command = answer[start:end]

            command, output = run_suggested_command(command)

            # Replace the command with the one that was actually run.
            answer = answer[:start] + command + answer[end:]
            executed = f"\n<response>{output}</response>"
        else:
            executed = ""

        messages.append({"role": "assistant", "content": answer})


if __name__ == '__main__':
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        pass
