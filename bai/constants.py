import os
from pathlib import Path


OPENAI_MODEL = "gpt-4-0125-preview"
ANTHROPIC_MODEL = "claude-3-opus-20240229"
ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
ANTHROPIC_MODEL = "claude-2.1"
USE_OPENAI = True

BASH_START = "<bash>"
BASH_END = "</bash>"

STYLE_SYSTEM = "\033[38;5;250m"
STYLE_ASSISTANT = "\033[38;5;39m"
STYLE_RESPONSE = "\033[38;5;208m"
STYLE_END = "\033[0m"

CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "bash-assistant"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

VI_MODE = True