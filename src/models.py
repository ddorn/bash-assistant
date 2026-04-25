"""Central LLM model configuration.

All models are specified as OpenRouter model IDs.
Browse the full list at: https://openrouter.ai/models

To change which model is used for a task, edit the constants below.
"""

_FAST_CHEAP = "openai/gpt-oss-120b:nitro"

# Bash assistant (bai)
BASH = _FAST_CHEAP

# Text fixer web UI
FIX = _FAST_CHEAP

# Tree book generator
TREE_BOOK = _FAST_CHEAP

# Git commit message generator
COMMIT = _FAST_CHEAP

# Speech transcript rewriter
REWRITE = _FAST_CHEAP

# Fallback for llms.py functions when no model is specified
DEFAULT = _FAST_CHEAP

# Models shown in web UI dropdowns
AVAILABLE = [
    "openai/gpt-oss-120b:nitro",
    "anthropic/claude-opus-4.7",
    "google/gemini-3.1-pro-preview",
]
