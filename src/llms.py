"""LLM interface via OpenRouter.

All text-generation calls go through a single OpenRouter client using
the OpenAI-compatible API. Model names are defined in models.py.
"""

import functools
import os
from typing import Generator, Iterator, TypeVar

import openai
from pydantic import BaseModel

import models

T = TypeVar("T", bound=BaseModel)


@functools.cache
def _client() -> openai.OpenAI:
    return openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def ai_chat(
    system: str | None,
    messages: list[dict],
    model: str | None = None,
    **kwargs,
) -> str:
    if model is None:
        model = models.DEFAULT
    if system:
        messages = [{"role": "system", "content": system}, *messages]
    kwargs.setdefault("temperature", 0.2)
    kwargs.setdefault("max_tokens", 1000)
    response = _client().chat.completions.create(model=model, messages=messages, **kwargs)
    return response.choices[0].message.content


def ai_stream(
    system: str | None,
    messages: list[dict],
    model: str | None = None,
    **kwargs,
) -> Generator[str, None, None]:
    if model is None:
        model = models.DEFAULT
    if system:
        messages = [{"role": "system", "content": system}, *messages]
    kwargs.setdefault("temperature", 1.0)
    for chunk in _client().chat.completions.create(
        model=model, messages=messages, stream=True, **kwargs
    ):
        text = chunk.choices[0].delta.content
        if text is not None:
            yield text


def ai_query(system: str, user: str, model: str | None = None) -> str:
    return ai_chat(system, [{"role": "user", "content": user}], model=model)


def ai_structured(
    system: str,
    user: str,
    output_format: type[T],
    model: str | None = None,
) -> T:
    if model is None:
        model = models.DEFAULT
    response = _client().beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=output_format,
    )
    return response.choices[0].message.parsed


def print_stream(stream: Iterator[str]) -> str:
    """Print a stream of text chunks and return the full text."""
    text = ""
    for chunk in stream:
        print(chunk, end="", flush=True)
        text += chunk
    return text
