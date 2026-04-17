from pathlib import Path
import sys
from textwrap import dedent
from typing import Iterator


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


def get_text_input(custom: str | None = "") -> str:
    """Get text input from the user, fallbacks to stdin if piped, or prompts the user."""

    import click

    if custom and len(custom) < 200 and Path(custom).is_file():
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


def print_join(iterable: Iterator[str]) -> str:
    """Print elements of an iterable as they come, and return the joined string."""

    text = ""
    for chunk in iterable:
        print(chunk, end="")
        text += chunk
    return text


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
