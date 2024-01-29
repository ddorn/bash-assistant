import os
from pathlib import Path


MODEL = "gpt-4-1106-preview"

BASH_START = "<bash>"
BASH_END = "</bash>"

STYLE_SYSTEM = "\033[38;5;250m"
STYLE_ASSISTANT = "\033[38;5;39m"
STYLE_RESPONSE = "\033[38;5;208m"
STYLE_END = "\033[0m"

CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "bash-assistant"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

VI_MODE = True