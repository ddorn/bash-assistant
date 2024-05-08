from pathlib import Path
from typing import Generator
import anthropic
from httpx import stream
import openai
import constants
import config


DATA = Path(__file__).parent.parent / "data"
DATA.mkdir(exist_ok=True, parents=True)


def fmt(
    text: str,
    fg: int | tuple[int, int, int] = None,
    bg: int | tuple[int, int, int] = None,
    underline: bool = False,
) -> str:
    """Format the text with the given colors."""

    mods = ""

    if underline:
        mods += "\033[4m"

    if fg is not None:
        if isinstance(fg, int):
            mods += f"\033[38;5;{fg}m"
        else:
            mods += f"\033[38;2;{fg[0]};{fg[1]};{fg[2]}m"

    if bg is not None:
        if isinstance(bg, int):
            mods += f"\033[48;5;{bg}m"
        else:
            mods += f"\033[48;2;{bg[0]};{bg[1]};{bg[2]}m"

    if mods:
        text = mods + text + "\033[0m"

    return text


def fmt_diff(diff: list[str]) -> tuple[str, str]:
    """Format the output of difflib.ndiff.

    Returns:
        tuple[str, str]: The two strings (past, new) with the differences highlighted in ANSI colors.
    """

    past = ""
    new = ""
    for line in diff:
        mark = line[0]
        line = line[2:]
        match mark:
            case " ":
                past += line
                new += line
            case "-":
                past += fmt(line, fg=1, underline=True)
            case "+":
                new += fmt(line, fg=2, underline=True)
            case "?":
                pass

    return past, new


def get_text_input(custom: str = "") -> str:
    """Get text input from the user, fallbacks to stdin if piped, or prompts the user."""

    import click

    if 0 < len(custom) < 200 and Path(custom).is_file():
        return Path(custom).read_text()

    if custom:
        return custom

    if not click.get_text_stream("stdin").isatty():
        # Read from stdin, if it is piped
        text = click.get_text_stream("stdin").read()
    else:
        # Prompt the user
        text = click.edit()

    if text is None:
        raise click.Abort()
    return text


anthropic_client = anthropic.Client(api_key=config.ANTHROPIC_API_KEY)


def ai_chat(system: str | None, messages: list[dict[str, str]], model: str = None) -> str:
    """Chat with the AI using the given messages."""

    if model is None:
        if constants.USE_OPENAI:
            model = constants.OPENAI_MODEL
        else:
            model = constants.ANTHROPIC_MODEL

    if "claude" in model:
        kwargs = {}
        if system:
            kwargs = dict(system=system)

        message = anthropic_client.messages.create(
            model=constants.ANTHROPIC_MODEL,
            max_tokens=1000,
            temperature=0.2,
            messages=messages,
            **kwargs,
        )
        return message.content
    else:
        if system:
            messages = [
                dict(role="system", content=system),
                *messages,
            ]
        response = openai.chat.completions.create(
            model=constants.OPENAI_MODEL,
            max_tokens=1000,
            temperature=0.2,
            messages=messages,
        )
        return response.choices[0].message.content


def ai_stream(
    system: str | None, messages: list[dict[str, str]], model: str = None, **kwargs
) -> Generator[str, None, None]:
    """Stream with the AI using the given messages."""

    if model is None:
        if constants.USE_OPENAI:
            model = constants.OPENAI_MODEL
        else:
            model = constants.ANTHROPIC_MODEL

    new_kwargs = dict(
        max_tokens=1000,
        temperature=0.2,
    )
    kwargs = {**new_kwargs, **kwargs}

    if "claude" in model:
        if system:
            kwargs["system"] = system

        with anthropic_client.messages.stream(
            model=constants.ANTHROPIC_MODEL,
            messages=messages,
            **kwargs,
        ) as stream:
            for text in stream.text_stream:
                yield text
    else:
        if system:
            messages = [
                dict(role="system", content=system),
                *messages,
            ]
        response = openai.chat.completions.create(
            model=constants.OPENAI_MODEL,
            messages=messages,
            stream=True,
            **kwargs,
        )

        for chunk in response:
            text = chunk.choices[0].delta.content
            if text is None:
                break
            yield text


def ai_query(system: str, user: str) -> str:
    """Query the AI with the given system and user message."""

    return ai_chat(system, [dict(role="user", content=user)])
