"""Central LLM model configuration.

All models are specified as OpenRouter model IDs.
Browse the full list at: https://openrouter.ai/models

To change which model is used for a task, edit the constants below.
"""

# General-purpose model for most tasks (bash assistant, translation, text fixing)
DEFAULT = "anthropic/claude-3.5-sonnet"

# Fast and cheap Claude model
FAST = "anthropic/claude-3-haiku"

# Cheapest viable model (good for structured outputs, commit messages, etc.)
CHEAPEST = "openai/gpt-4o-mini"

# Cheap but capable enough for most tasks
CHEAP_BUT_GOOD = "openai/gpt-4o-mini"

# Model used for generating git commit messages
COMMIT = "openai/gpt-4o-mini"

# Model used for rewriting speech transcripts
REWRITE = "anthropic/claude-3-haiku"

# Models shown in web UI dropdowns
AVAILABLE = [
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-haiku",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "google/gemini-flash-1.5",
]
