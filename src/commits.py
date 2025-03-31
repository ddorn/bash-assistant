import re
import subprocess
from pathlib import Path

import openai
import typer
from pydantic import BaseModel

import constants
import utils


class CommitModel(BaseModel):
    description: str
    breaking: str | None
    title: str

    def to_string(self):
        description = self.description
        title = self.title
        if self.breaking:
            description += "\n\nBREAKING CHANGE: " + self.breaking.strip()
            # Add a ! after the type in the title. Title fmt: type(optional scope): title
            # Add the ! after the optional scope.
            title = re.sub(r"(\w+)(\(.+?\))?:", r"\1\2!:", title)

        return f"{title}\n\n{description.strip()}"


app = typer.Typer(add_completion=False, no_args_is_help=True)

SYSTEM = """Generate a commit message for the following git diff. Use the Conventional Commits format.

Template:
{{
    "description": "List the changes in details and why they were made.",
    "breaking": "optional: what breaking changes were made",
    "title": "type(optional scope): commit title of up to 50 characters"
}}

Example output:
{{
    "description": "* Improve timer command to use multiple durations and ring options.\n* Add dependency on rich for better output formatting.",
    "breaking": "The timer now accepts a list instead of a single string for its \"duration\" argument.",
    "title": "feat(timer): Add multiple durations and ring options"
}}

Example output:
{{
    "description": "Mypy now properly emulates attrs' logic so that custom `__hash__` implementations are preserved, `@frozen` subclasses are always hashable, and classes are only made unhashable based on the values of `eq` and `unsafe_hash`.",
    "breaking": null,
    "title": "fix(attrs): Fix emulating hash method logic"
}}

Example output:
{{
    "description": "The logs provide a more reliable source for evaluating the safeguard's response than the model output. Previously, we expected the answer to remain unchanged when checks passed, but observed differences in practice. This commit bases the evaluation on activated strategies recorded in logs, though the precise conditions triggering each strategy require further investigation.",
    "breaking": null,
    "title": "fix(nemo): Use logs for safeguard evaluation"
}}

Last commits (provided for context, but may not be well written):
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
    else:
        commits = last_commits.split("-----")
        formated_commits = []
        for commit in commits:
            lines = commit.strip().splitlines()
            if not lines:
                continue  # for the last ----- separator
            title = lines.pop(0)
            breaking_changes = "\n".join(
                line for line in lines if line.startswith("BREAKING CHANGE:")
            )
            body = "\n".join(
                line for line in lines if line and not line.startswith("BREAKING CHANGE:")
            )
            if not breaking_changes.strip():
                breaking_changes = None

            commit = CommitModel(description=body, breaking=breaking_changes, title=title)
            formated_commits.append(commit.model_dump_json(indent=4))
        last_commits = "\n".join(formated_commits)

    if not additional_guidelines:
        additional_guidelines = "No additional guidelines."

    system = SYSTEM.format(last_commits=last_commits, additional_guidelines=additional_guidelines)
    diff += "\n\n\nGenerate a commit message for the previous git diff."

    messages = [
        dict(role="system", content=system),
        dict(role="user", content=diff),
        # dict(role="assistant", content="<commit>"),
    ]

    if not utils.confirm_cost(messages, model, max_cost):
        return

    answer = (
        openai.beta.chat.completions.parse(
            messages=messages,
            model=model,
            response_format=CommitModel,
        )
        .choices[0]
        .message
    )
    commit = answer.parsed

    if commit is None:
        print("Generated (invalid) JSON:", answer.content)
        exit(1)

    message = commit.to_string()

    if commit_file:
        commit_file.write_text(message + "\n" + commit_file.read_text())
    else:
        # Print in yellow to make it stand out.
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
