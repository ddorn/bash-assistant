import subprocess

from openai import OpenAI

import prompt_toolkit as pt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.shell import BashLexer


VI_MODE = True
BASH_START = "<bash>"
BASH_END = "</bash>"

STYLE_SYSTEM = "\033[38;5;240m"
STYLE_ASSISTANT = "\033[38;5;39m"
STYLE_RESPONSE = "\033[38;5;208m"
STYLE_END = "\033[0m"

SYSTEM = f"""
You are being run in a scafold on an archlinux machine running bash.
You can run bash commands at the end of your answer by wrapping it in {BASH_START} and {BASH_END} tags.
You will be shown the output of the command, but you cannot interact with it.
Your answers are concise. You don't ask the user before running a command, just run it. Don't provide explanations unless asked.
""".strip()

def main():
    client = OpenAI()

    messages = [{"role": "system", "content": SYSTEM}]

    console = pt.PromptSession(
        message="\n> ",
        history=FileHistory("history.txt"),
        auto_suggest=AutoSuggestFromHistory(),
        vi_mode=VI_MODE,
    )
    bash_console = pt.PromptSession(
        message="run: ",
        history=FileHistory("commands.txt"),
        auto_suggest=AutoSuggestFromHistory(),
        vi_mode=VI_MODE,
        lexer=PygmentsLexer(BashLexer),
    )


    # Print the system message.
    print(f"{STYLE_SYSTEM}System: {SYSTEM}{STYLE_END}")

    while True:
        question = console.prompt()
        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            stream=True,
        )

        answer = ""
        print(STYLE_ASSISTANT + "Assistant: ", end="")
        for chunk in response:
            text = chunk.choices[0].delta.content
            if text is None:
                continue

            answer += text
            # Show the answer as soon as it's ready.
            print(text, end="")

            # Check if the answer has a bash command.
            if BASH_START in answer:
                # Find the matching opening tags
                start = answer.find(BASH_START)
                end = answer.find(BASH_END, start)
                if end == -1 or end < start:
                    continue

                # Extract the command
                command = answer[start + len(BASH_START):end]

                # Ask to run the command. Print with highlighting.
                try:
                    print()
                    to_run = bash_console.prompt(default=command)
                except KeyboardInterrupt:
                    output = "canceled"
                else:
                    # Run the command.
                    output = subprocess.check_output(to_run, shell=True)
                    output = output.decode("utf-8")

                print(f"{STYLE_RESPONSE}{output}{STYLE_END}")
                answer += f"\n<response>{output}</response>"
                break

        messages.append({"role": "assistant", "content": answer})

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
