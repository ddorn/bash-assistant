from dataclasses import dataclass
from pathlib import Path
import sys
from textwrap import dedent
from typing import Generator, Iterator
import anthropic
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


def fmt_diff(diff: Iterator[str]) -> tuple[str, str]:
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


def ai_chat(
    system: str | None, messages: list[dict[str, str]], model: str = None, confirm: bool = False
) -> str:
    """Chat with the AI using the given messages."""

    if model is None:
        model = constants.OPENAI_MODEL if constants.USE_OPENAI else constants.ANTHROPIC_MODEL

    if system:
        messages = [dict(role="system", content=system)] + messages

    if confirm:
        estimation = estimate_cost(messages, model)

        if not confirm_action(f"{model}: {estimation}. Confirm?"):
            return "Aborted."

    if "claude" in model:
        # System message is a kwarg
        if system:
            del messages[0]

        message = anthropic_client.messages.create(
            model=constants.ANTHROPIC_MODEL,
            max_tokens=1000,
            temperature=0.2,
            system=system,
            messages=messages,
        )
        return message.content[0].text
    else:
        response = openai.chat.completions.create(
            model=constants.OPENAI_MODEL,
            max_tokens=1000,
            temperature=0.2,
            messages=messages,
        )
        return response.choices[0].message.content


def ai_stream(
    system: str | None,
    messages: list[dict[str, str]],
    model: str = None,
    confirm: float | None = None,
    **kwargs,
) -> Generator[str, None, None]:
    """Stream with the AI using the given messages."""

    if model is None:
        model = constants.OPENAI_MODEL if constants.USE_OPENAI else constants.ANTHROPIC_MODEL

    if system:
        messages = [dict(role="system", content=system)] + messages

    if confirm is not None:
        estimation = estimate_cost(messages, model)
        msg = f"{model}: {estimation}"

        if estimation.input_cost < confirm:
            print(f"{msg}. Confirming automatically.")
        elif not confirm_action(f"{msg}. Confirm?"):
            return

    new_kwargs = dict(
        max_tokens=1000,
        temperature=0.2,
    )
    kwargs = {**new_kwargs, **kwargs}

    if "claude" in model:
        if system:
            del messages[0]

        if messages[-1]["role"] == "assistant":
            yield messages[-1]["content"]

        with anthropic_client.messages.stream(
            model=constants.ANTHROPIC_MODEL,
            messages=messages,
            system=system,
            **kwargs,
        ) as stream:
            for text in stream.text_stream:
                yield text
    else:
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


def print_join(iterable: Iterator[str]) -> str:
    """Print elements of an iterable as they come, and return the joined string."""

    text = ""
    for chunk in iterable:
        print(chunk, end="")
        text += chunk
    return text


def ai_query(system: str, user: str, model: str | None = None, confirm: bool = False) -> str:
    """Query the AI with the given system and user message."""

    return ai_chat(system, [dict(role="user", content=user)], model=model, confirm=confirm)


def soft_parse_xml(text: str) -> dict[str, str | dict | list]:
    """Extract xml tags from the free form text."""

    import re

    tags = re.findall(r"<(\w+)>(.*?)</\1>", text, re.DOTALL)
    map = {}
    for key, value in tags:
        value = dedent(value).strip()
        children = soft_parse_xml(value)
        value = children if children else value
        if key not in map:
            map[key] = value
        elif isinstance(map[key], list):
            map[key].append(value)
        else:
            map[key] = [map[key], value]

    return map


if __name__ == "__main__":
    text = """
    Salut <name>John</name>! <age>25</age> years old.

    Nested:
    <nested>
        Okay
        <key>value1</key>
        <key>value2</key>
        <other><with>nesting</with> yes?</other>
    </nested>

    Thanks <name>Diego</name>!
    """

    xml = soft_parse_xml(text)
    assert xml == dict(
        name=["John", "Diego"],
        age="25",
        nested=dict(
            key=["value1", "value2"],
            other={"with": "nesting"},
        ),
    )
    print("XML test passed! 🎉")


@dataclass
class CostEstimation:
    input_tokens: int
    input_cost: float
    output_cost: float
    is_approximate: bool

    def __str__(self):
        return (
            f"Cost for {self.input_tokens} tokens: "
            f"{self.input_cost:.4f}$ + "
            f"{self.output_cost * 1000:.4f}$/1k output tokens"
        )


def estimate_cost(messages: list[dict], model: str) -> CostEstimation:
    """Estimate the cost of the AI completion."""

    import tiktoken

    input_cost, output_cost = constants.MODELS_COSTS[model]

    try:
        encoding = tiktoken.encoding_for_model(model)
        approx = False
    except KeyError:
        # This makes it work also for Anthropic models.
        # This will be less accurate than for OpenAI, but their tokenization is not public
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        approx = True

    input_tokens = 0
    for msg in messages:
        input_tokens += len(encoding.encode(msg["content"]))
        input_tokens += 4  # for the role and the separator

    return CostEstimation(
        input_tokens=input_tokens,
        input_cost=input_cost * input_tokens / 1_000_000,
        output_cost=output_cost / 1_000_000,
        is_approximate=approx,
    )


def notify(title: str, message: str, urgency: str = "normal"):
    """Send a desktop notification."""

    import subprocess

    subprocess.run(["notify-send", title, message, f"--urgency={urgency}"], check=True)


def confirm_action(message: str) -> bool:
    """Ask the user to confirm an action."""

    from InquirerPy import inquirer

    try:
        return inquirer.confirm(message).execute()
    except EOFError:
        # When using `d commit` through git, the confirmation doesn't work, as the input is not
        # Interactive. In this case, we fallback to a simple print and can't confirm.
        print(message + " No.", file=sys.stderr)
        print("Could not confirm action. Aborted.", file=sys.stderr)
        return False
