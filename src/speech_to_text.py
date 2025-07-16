#!/usr/bin/env python

"""
An interactive audio recorder and transcriber with an optional rewrite step.

Records audio and transcribes it using a selected provider (Groq or OpenAI).
- In interactive mode, it shows a live UI to control recording and options.
- In non-interactive mode, it transcribes a given audio file.
- Optionally rewrites the transcript for clarity and formatting using an LLM.
"""
import os
import shutil
import subprocess
import tempfile
import time
import json
import signal
from dataclasses import dataclass, field
from pathlib import Path

import typer
from rich.console import Console
from rich.text import Text
import platformdirs

from textual.app import App, ComposeResult
from textual.widgets import Footer, Static

from dotenv import load_dotenv

load_dotenv()


# --- Configuration ---
TRANSCRIPTION_PROVIDERS = {
    "groq/whisper-large-v3": {"model": "groq/whisper-large-v3", "api_key_env": "GROQ_API_KEY"},
    "openai/whisper-1": {"model": "openai/whisper-1", "api_key_env": "OPENAI_API_KEY"},
}

REWRITE_PROVIDERS = {
    "groq/llama3-70b": {
        "model": "groq/llama3-70b-8192",
        "api_key_env": "GROQ_API_KEY",
    },
    "claude-3.5-haiku": {
        "model": "anthropic/claude-3-5-haiku-20241022",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "gpt-4o": {
        "model": "openai/gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
    },
}


# --- Settings Management ---
def get_settings_path() -> Path:
    """Returns the path to the settings file."""
    settings_dir = platformdirs.user_config_path("2text", ensure_exists=True)
    return settings_dir / "settings.json"


def load_settings() -> dict:
    """Loads settings from the JSON file, returning an empty dict on failure."""
    settings_path = get_settings_path()
    if not settings_path.exists():
        return {}
    try:
        with open(settings_path, "r") as f:
            # Handle empty file case
            content = f.read()
            if not content:
                return {}
            settings = json.loads(content)
            # Validate provider settings
            if "provider" in settings and settings["provider"] not in TRANSCRIPTION_PROVIDERS:
                del settings["provider"]
            if (
                "rewrite_provider" in settings
                and settings["rewrite_provider"] not in REWRITE_PROVIDERS
            ):
                del settings["rewrite_provider"]
            return settings
    except (json.JSONDecodeError, IOError):
        return {}


def save_settings(state: "AppState"):
    """Saves relevant settings from the app state to the JSON file."""
    settings_path = get_settings_path()
    settings_to_save = {
        "provider": state.provider,
        "language": state.language,
        "rewrite_provider": state.rewrite_provider,
    }
    with open(settings_path, "w") as f:
        json.dump(settings_to_save, f, indent=2)


# --- State Management ---
@dataclass
class AppState:
    """Holds the dynamic state of the application."""

    provider: str = next(iter(TRANSCRIPTION_PROVIDERS.keys()))
    rewrite_provider: str = next(iter(REWRITE_PROVIDERS.keys()))
    language: str | None = None
    rewrite: bool = False
    output_file: Path | None = None
    segment_start_time: float = field(default_factory=time.time)
    is_paused: bool = False
    total_recorded_duration: float = 0.0

    @property
    def model_name(self) -> str:
        """Returns the transcription model name based on the provider."""
        return TRANSCRIPTION_PROVIDERS[self.provider]["model"]

    def get_duration(self) -> str:
        """Returns the formatted recording duration."""
        current_segment_duration = 0.0
        if not self.is_paused:
            current_segment_duration = time.time() - self.segment_start_time
        total_duration = self.total_recorded_duration + current_segment_duration
        return f"{total_duration:.1f}s"

    def get_file_size_str(self) -> str:
        """Returns the formatted file size as a string."""
        if self.output_file and self.output_file.exists():
            size_bytes = self.output_file.stat().st_size
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            else:
                return f"{size_bytes / (1024 * 1024):.1f} MB"
        return "0 B"


class RecordingError(Exception):
    """Exception raised when the recording process fails."""


# --- Textual Application for Interactive Mode ---
LANGUAGES = ["en", "fr", None]  # None is for auto-detection


class TranscriberApp(App):
    """A Textual app for interactive recording."""

    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        ("space", "toggle_pause", "Pause"),
        ("t", "transcribe(False)", "Transcribe"),
        ("r", "transcribe(True)", "Rewrite"),
        ("l", "toggle_language", ""),
        ("p", "toggle_provider", ""),
        ("s", "toggle_rewrite_provider", ""),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.recording_process: subprocess.Popen | None = None
        self.ffmpeg_log_handle = None

    def compose(self) -> ComposeResult:
        yield Static(id="info_file")
        yield Static(id="info_stats")
        yield Static(id="info_provider")
        yield Static(id="info_rewrite_provider")
        yield Static(id="info_language")
        yield Footer()

    def on_mount(self) -> None:
        """Start the recording, UI timer, and background import of litellm."""
        # Start a background worker to import the slow module.
        self.run_worker(self.import_litellm, exclusive=True, thread=True)

        # Set the static file path line
        file_info_widget = self.query_one("#info_file", Static)
        file_info_text = Text("Recording to ")
        file_info_text.append(str(self.state.output_file.resolve()), style="cyan")
        file_info_widget.update(file_info_text)

        # Set the initial provider line and start the stats timer
        self.update_provider_display()
        self.update_rewrite_provider_display()
        self.update_language_display()
        self.update_timer = self.set_interval(1 / 10, self.update_stats)
        self.start_recording()

    def import_litellm(self) -> None:
        """A worker method to import litellm in the background."""
        import litellm as litellm

    def update_stats(self) -> None:
        """Update the stats display and check for recording process failure."""
        if self.recording_process and self.recording_process.poll() is not None:
            # The recording process has terminated unexpectedly.
            if self.ffmpeg_log_handle:
                self.ffmpeg_log_handle.close()
                self.ffmpeg_log_handle = None

            log_path = self.state.output_file.with_suffix(".log")
            error_message = f"Recording process failed. See log: {log_path}"
            self.exit(RecordingError(error_message))

        stats_widget = self.query_one("#info_stats", Static)
        icon = "⏸️ " if self.state.is_paused else "🔴"
        label = "Paused" if self.state.is_paused else "Rec"
        stats_text = Text(
            f"{icon} {label}: {self.state.get_duration()} | {self.state.get_file_size_str()}"
        )
        stats_widget.update(stats_text)

    def update_provider_display(self) -> None:
        """Update the provider display."""
        provider_widget = self.query_one("#info_provider", Static)
        markup = f" [bold #FFA62B]p[/] Audio Provider: [green]{self.state.provider}[/]"
        provider_widget.update(Text.from_markup(markup))

    def update_rewrite_provider_display(self) -> None:
        """Update the rewrite provider display."""
        provider_widget = self.query_one("#info_rewrite_provider", Static)
        markup = f" [bold #FFA62B]s[/] Rewrite Provider: [magenta]{self.state.rewrite_provider}[/]"
        provider_widget.update(Text.from_markup(markup))

    def update_language_display(self) -> None:
        """Update the language display."""
        language_widget = self.query_one("#info_language", Static)
        lang_text = self.state.language if self.state.language else "auto"
        markup = f" [bold #FFA62B]l[/] Language: [blue]{lang_text}[/]"
        language_widget.update(Text.from_markup(markup))

    def start_recording(self):
        """Starts the ffmpeg recording process, capturing stderr for error reporting."""
        log_path = self.state.output_file.with_suffix(".log")
        self.ffmpeg_log_handle = log_path.open("w", encoding="utf-8")
        ffmpeg_cmd = [
            "ffmpeg",
            "-f",
            "alsa",
            "-i",
            "default",
            "-acodec",
            "libmp3lame",
            "-ab",
            "128k",
            "-y",
            str(self.state.output_file),
        ]
        # Capture stderr to detect errors if the process fails during recording
        self.recording_process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=self.ffmpeg_log_handle,
            stderr=self.ffmpeg_log_handle,
        )

    def stop_recording(self):
        """Stops the ffmpeg recording process."""
        if self.recording_process and self.recording_process.poll() is None:
            if self.state.is_paused:
                # Resume before terminating to ensure it exits cleanly
                os.kill(self.recording_process.pid, signal.SIGCONT)
                self.state.is_paused = False

            self.recording_process.terminate()

            # We just wait for ffmpeg to finish writing the file.
            self.recording_process.wait(timeout=15)
            if self.recording_process.poll() is None:  # Force kill if terminate fails
                self.recording_process.kill()

        if self.ffmpeg_log_handle:
            self.ffmpeg_log_handle.close()
            self.ffmpeg_log_handle = None

    def action_transcribe(self, rewrite: bool) -> None:
        """Stop recording and exit the app to transcribe."""
        # If we are actively recording, add the final segment's duration
        if not self.state.is_paused:
            segment_duration = time.time() - self.state.segment_start_time
            self.state.total_recorded_duration += segment_duration

        self.state.rewrite = rewrite
        self.stop_recording()
        self.exit(self.state)

    def action_toggle_pause(self) -> None:
        """Toggles the recording between paused and active states."""
        if not self.recording_process or self.recording_process.poll() is not None:
            return  # Can't pause/resume if recording isn't active

        self.state.is_paused = not self.state.is_paused

        if self.state.is_paused:
            # Pause the recording
            os.kill(self.recording_process.pid, signal.SIGSTOP)
            segment_duration = time.time() - self.state.segment_start_time
            self.state.total_recorded_duration += segment_duration
        else:
            # Resume the recording
            self.state.segment_start_time = time.time()
            os.kill(self.recording_process.pid, signal.SIGCONT)
        self.update_stats()

    def action_toggle_provider(self) -> None:
        """Toggle the transcription provider."""
        providers = list(TRANSCRIPTION_PROVIDERS.keys())
        current_index = providers.index(self.state.provider)
        next_index = (current_index + 1) % len(providers)
        self.state.provider = providers[next_index]
        self.update_provider_display()

    def action_toggle_rewrite_provider(self) -> None:
        """Toggle the rewrite provider."""
        providers = list(REWRITE_PROVIDERS.keys())
        current_index = providers.index(self.state.rewrite_provider)
        next_index = (current_index + 1) % len(providers)
        self.state.rewrite_provider = providers[next_index]
        self.update_rewrite_provider_display()

    def action_toggle_language(self) -> None:
        """Toggle the transcription language."""
        try:
            current_index = LANGUAGES.index(self.state.language)
            next_index = (current_index + 1) % len(LANGUAGES)
        except ValueError:
            next_index = 0  # Default to first language if current is not in list
        self.state.language = LANGUAGES[next_index]
        self.update_language_display()

    def action_quit(self) -> None:
        """Quit the application without transcribing."""
        self.stop_recording()
        self.exit(None)


# --- Core Logic ---
def copy_to_clipboard(text: str):
    """Copies the given text to the system clipboard."""
    if os.environ.get("WAYLAND_DISPLAY"):
        cmd = ["wl-copy"]
    else:
        cmd = ["xclip", "-selection", "clipboard"]
    subprocess.run(cmd, input=text, encoding="utf-8")


def rewrite_transcript(console: Console, transcript: str, model_name: str) -> str:
    """Rewrites the transcript using an LLM for clarity and formatting."""
    # litellm is slow to import at the start of the script, so we delayed it
    from litellm import completion

    console.print("✍️  Rewriting transcript...", end="")
    rewrite_start = time.time()
    system_prompt = """You are an expert editor. Your sole task is to silently correct the text between <transcript> and </transcript>.

Fix any transcription errors, punctuation, and capitalization. Format it into clean paragraphs.

**Only output the corrected text.** Do not add comments, explanations, or any meta-text.
The output should be a clean, corrected version of the original text, ready to be copied and pasted directly."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": transcript},
    ]

    response = completion(
        model=model_name,
        messages=messages,
    )
    rewritten_text = response.choices[0].message.content.strip()
    rewrite_duration = time.time() - rewrite_start
    console.print(f" {rewrite_duration:.2f}s ✓")
    return rewritten_text


def transcribe_audio(console: Console, file_path: Path, state: AppState):
    """Handles the transcription process and prints the result."""
    # litellm is slow to import at the start of the script, so we delayed it
    from litellm import transcription

    console.print(f"🎤 Transcribing with [bold]{state.provider}[/bold]...", end="")
    transcribe_start = time.time()

    kwargs = {"model": state.model_name}
    if state.language:
        kwargs["language"] = state.language

    with open(file_path, "rb") as audio_file:
        response = transcription(
            file=audio_file,
            **kwargs,
        )
    text = response["text"].strip()
    transcribe_duration = time.time() - transcribe_start
    console.print(f" {transcribe_duration:.2f}s ✓")

    if state.rewrite:
        rewrite_provider_key = state.rewrite_provider
        rewrite_provider_config = REWRITE_PROVIDERS[rewrite_provider_key]
        model_for_rewrite = rewrite_provider_config["model"]

        text = rewrite_transcript(console, text, model_for_rewrite)

    console.print(f"\n[bold]Transcript:[/bold]\n{text}")

    copy_to_clipboard(text)
    console.print("\n[dim]Transcript copied to clipboard.[/dim]")


def check_dependencies(console: Console):
    """Checks for required tools and filters providers based on available API keys."""
    global TRANSCRIPTION_PROVIDERS, REWRITE_PROVIDERS

    # Check for ffmpeg
    if not shutil.which("ffmpeg"):
        console.print("[bold red]Error: ffmpeg not found.[/bold red]")
        console.print("Please install ffmpeg and ensure it's in your system's PATH.")
        console.print("  - On Debian/Ubuntu: sudo apt update && sudo apt install ffmpeg")
        console.print("  - On macOS (with Homebrew): brew install ffmpeg")
        raise typer.Exit(code=1)

    # Dynamically filter providers based on available API keys
    def filter_providers(providers: dict) -> dict:
        """Removes providers for which the API key is not set."""
        return {
            key: config
            for key, config in providers.items()
            if os.environ.get(config["api_key_env"])
        }

    transcription_keys = set(config["api_key_env"] for config in TRANSCRIPTION_PROVIDERS.values())
    rewrite_keys = set(config["api_key_env"] for config in REWRITE_PROVIDERS.values())
    TRANSCRIPTION_PROVIDERS = filter_providers(TRANSCRIPTION_PROVIDERS)
    REWRITE_PROVIDERS = filter_providers(REWRITE_PROVIDERS)

    if not TRANSCRIPTION_PROVIDERS:
        console.print("[bold red]Error: No transcription providers could be configured.[/bold red]")
        console.print(f"Please set at least one API key ({', '.join(transcription_keys)}).")
        raise typer.Exit(1)
    if not REWRITE_PROVIDERS:
        console.print("[bold red]Error: No rewrite providers could be configured.[/bold red]")
        console.print(f"Please set at least one API key ({', '.join(rewrite_keys)}).")
        raise typer.Exit(1)


# --- Main Application ---
def main(
    file: str | None = typer.Argument(None, help="Path to an audio file to transcribe directly."),
    language: str = typer.Option(
        None,
        "-l",
        "--language",
        help="Language of the audio. If not provided, the language will be detected automatically.",
    ),
    provider: str = typer.Option(
        None, "--provider", help="API provider to use. Overrides saved setting."
    ),
    rewrite: bool = typer.Option(
        False, "--rewrite", help="Rewrite the transcript for clarity (non-interactive mode only)."
    ),
):
    """
    An interactive audio recorder and transcriber.
    Run without arguments to start an interactive recording session.
    """
    console = Console()
    check_dependencies(console)

    temp_dir = Path(tempfile.gettempdir())

    # --- Non-Interactive (File-based) Mode ---
    if file:
        # Use dataclass defaults, overridden only by CLI args. No config file.
        state = AppState(rewrite=rewrite)
        if provider:
            state.provider = provider
        if language:
            state.language = language

        if file.upper() == "LAST":
            recordings = sorted(temp_dir.glob("recording-*.mp3"))
            if not recordings:
                console.print(f"[red]Error: No 'LAST' recording found in {temp_dir}.[/red]")
                raise typer.Exit(1)
            audio_file_path = recordings[-1]
            console.print(f"Using last recording: [cyan]{audio_file_path}[/cyan]")
        else:
            audio_file_path = Path(file)

        if not audio_file_path.exists():
            console.print(f"[red]Error: File not found: {audio_file_path}[/red]")
            raise typer.Exit(1)

        transcribe_audio(console, audio_file_path, state)
        return

    # --- Interactive Mode ---
    # Load saved settings and override with any CLI args
    saved_settings = load_settings()
    if provider:
        saved_settings["provider"] = provider
    if language:
        saved_settings["language"] = language
    state = AppState(**saved_settings)
    state.output_file = temp_dir / f"recording-{time.strftime('%Y-%m-%d-%H-%M-%S')}.mp3"

    app = TranscriberApp(state)
    final_state = app.run()

    if isinstance(final_state, RecordingError):
        console.print(final_state, markup=False)
        console.print("[bold red]❌ Recording failed unexpectedly.[/bold red]")
        raise typer.Exit(1)

    elif final_state:
        console.print(f"✅ Recording finished. File: [cyan]{final_state.output_file}[/cyan]")
        save_settings(final_state)
        transcribe_audio(console, final_state.output_file, final_state)
    else:
        console.print(f"❌ Recording cancelled. File: [cyan]{state.output_file}[/cyan]")


if __name__ == "__main__":
    typer.run(main)
