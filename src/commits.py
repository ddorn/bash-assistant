import re
import subprocess
from pathlib import Path
from textwrap import dedent

import typer

import constants
from utils import (
    ai_stream,
    print_join,
    soft_parse_xml,
)


app = typer.Typer(add_completion=False, no_args_is_help=True)

SYSTEM = """Generate a commit message for the following git diff. Use the Conventional Commits format.

Template:
<commit>
    <description>
        optional body: describe the changes in more details
    </description>
    <breaking>
        optional scope: breaking changes to the interface etc.
    </breaking>
    <title>
        type(optional scope): title of up to 50 characters
    </title>
</commit>

Example output:
<commit>
    <description>
        Improve timer command to use multiple durations and ring options.
        Add dependency on rich for better output formatting.
    </description>
    <breaking>
        The timer now accepts a list instead of a single string.
    </breaking>
    <title>
        feat(timer): Add multiple durations and ring options
    </title>
</commit>

Example output:
<commit>
    <description>
        Mypy now properly emulates attrs' logic so that custom `__hash__`
        implementations are preserved, `@frozen` subclasses are always hashable,
        and classes are only made unhashable based on the values of `eq` and
        `unsafe_hash`.
    </description>
    <title>
        fix(attrs): Fix emulating hash method logic
    </title>
</commit>

Example output:
<commit>
    <description>
        * 📝 Add source examples for custom parameter types with Annotated
        * 📝 Update docs for custom parameters with Annotated
    </description>
    <title>
        docs: Update docs examples for custom param types using Annotated
    </title>
</commit>

Last commits:
{last_commits}

Additional guidelines for this codebase:
{additional_guidelines}"""


@app.command()
def commit(
    max_cost: float = 0.02,
    commit_file: Path = None,
    model: str = constants.OPENAI_MODEL,
    additional_guidelines: str = "",
):
    """Generate a commit message for the current changes."""

    if commit_file and commit_file.exists():
        # Check if there is already a commit message in the file.
        message = commit_file.read_text()
        message = message.partition("-------- >8 --------")[0]
        message = "\n".join(
            line for line in message.splitlines() if not line.strip().startswith("#")
        ).strip()

        if message:
            print("❌ There is already a commit message.")
            return

    # We probably want to see the status before committing.
    subprocess.run(["git", "status"])

    # Capture the diff for the changes that were already added to the index.
    diff = subprocess.run(
        ["git", "diff", "--cached"], capture_output=True, text=True, check=True
    ).stdout

    if not diff:
        print("❌ Nothing added to commit.")
        exit(1)

    last_commits = subprocess.run(
        ["git", "log", "--format=%B-----", "-n", "8"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if not last_commits:
        last_commits = "No commits yet."

    if not additional_guidelines:
        additional_guidelines = "No additional guidelines."

    system = SYSTEM.format(last_commits=last_commits, additional_guidelines=additional_guidelines)

    diff += "\n\n\nGenerate a commit message for the previous git diff."

    messages = [
        dict(role="user", content=diff),
        # dict(role="assistant", content="<commit>"),
    ]

    response = print_join(ai_stream(system, messages, model=model, confirm=max_cost))
    tags = soft_parse_xml(response).get("commit", {})

    description = tags.get("description", "")
    breaking = tags.get("breaking", "")
    title = tags.get("title", "")

    if not title:
        print("❌ No commit message generated.")
        exit(1)

    if breaking:
        description += "\n\nBREAKING CHANGE: " + breaking
        # Add a ! after the type in the title. Title fmt: type(optional scope): title
        # Add the ! after the optional scope.
        title = re.sub(r"(\w+)(\(.+?\))?:", r"\1\2!:", title)

    message = f"{title}\n\n{description.strip()}"

    # Print in yellow to make it stand out.
    if commit_file:
        commit_file.write_text(message)
    else:
        print(f"\n\033[33m{message}\033[0m")


@app.command()
def commit_install(force: bool = False):
    """Install the commit hook to generate commit messages."""

    hook = constants.ROOT / "scripts" / "prepare-commit-msg"
    git_path = subprocess.run(
        ["git", "rev-parse", "--git-dir"], capture_output=True, text=True, check=True
    ).stdout.strip()
    hook_path = Path(git_path) / "hooks" / "prepare-commit-msg"

    if hook_path.exists() and not force:
        print(f"❌ {hook_path} already exists. Use --force to overwrite.")
        quit(1)

    subprocess.run(["cp", hook, hook_path], check=True)
    print("✅ Commit hook installed.")


if __name__ == "__main__":
    app()
