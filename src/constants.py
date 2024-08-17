import os
from pathlib import Path


OPENAI_MODEL = "gpt-4o"
ANTHROPIC_MODEL = "claude-3-opus-20240229"
ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
ANTHROPIC_MODEL = "claude-2.1"
CHEAPEST_MODEL = "gpt-4o-mini"
CHEAP_BUT_GOOD = "gpt-4o-mini"
MODELS_COSTS = {
    "claude-3-5-sonnet-20240620": (3, 15),
    "gpt-4o": (5, 15),
    "gpt-4-turbo": (10, 30),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-3.5-turbo": (0.5, 1.5),
    "claude-3-opus-20240229": (15, 75),
    "claude-3-sonnet-20240229": (3, 15),
    "claude-3-haiku-20240229": (0.25, 1.25),
}
MODELS = list(MODELS_COSTS)
USE_OPENAI = True

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
