import os
from pathlib import Path


BASH_START = "<bash>"
BASH_END = "</bash>"

STYLE_SYSTEM = "\033[38;5;250m"
STYLE_ASSISTANT = "\033[38;5;39m"
STYLE_RESPONSE = "\033[38;5;208m"
STYLE_END = "\033[0m"

CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "bash-assistant"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
PROMPTS_FILE = SRC / "prompts.yaml"

VI_MODE = True
