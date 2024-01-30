import openai
from constants import MODEL


def fmt(text: str,
        fg: int | tuple[int, int, int] = None,
        bg: int | tuple[int, int, int] = None,
        underline: bool = False) -> str:
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
        tuple[str, str]: The two strings (past, new) with the differences highlighted in ANSI colors."""

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


def ai_query(system: str, user: str) -> str:
    """Query the AI with the given system and user message."""

    response = openai.chat.completions.create(
        messages=[
            dict(role="system", content=system),
            dict(role="user", content=user),
        ],
        model=MODEL,
    )

    return response.choices[0].message.content


def get_text_input(custom: str = "") -> str:
    """Get text input from the user, fallbacks to stdin if piped, or prompts the user."""

    import click

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
