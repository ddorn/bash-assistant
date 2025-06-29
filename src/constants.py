import os
from pathlib import Path


OPENAI_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_MODEL = "claude-3-opus-20240229"
ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
ANTHROPIC_MODEL = "claude-2.1"
CHEAPEST_MODEL = "gpt-4o-mini"
CHEAP_BUT_GOOD = "gpt-4o-mini"
MODELS_COSTS = {
    "claude-3-7-sonnet-latest": (3, 15),
    "claude-3-5-haiku-latest": (0.8, 4),
    "claude-3-5-sonnet-latest": (3, 15),
    "o3-mini": (1.10, 4.40),
    "gpt-4o": (5, 15),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4-turbo": (10, 30),
    "claude-3-opus-20240229": (15, 75),
    "claude-3-haiku-20240229": (0.25, 1.25),
    "claude-sonnet-4-20250514": (3, 15),
}
OLDER_MODELS_COSTS = {
    "gpt-3.5-turbo": (0.5, 1.5),
    "gpt-4o-2024-08-06": (5, 15),
    "claude-3-sonnet-20240229": (3, 15),
}
ALL_MODELS_COSTS = {**MODELS_COSTS, **OLDER_MODELS_COSTS}

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
