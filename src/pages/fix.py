from html import escape
import difflib
import re
from textwrap import dedent
import streamlit as st

from utils import ai_stream


with st.container(border=True):
    text = st.text_area("Text to fix")

    system_prompts = {
        "Default": """
            You are given a text and you need to fix the language (typos, grammar, ...).
            If needed, fix the formatting and, when relevant, ensure the text is inclusive.
            Output directly the corrected text, without any comment.
            """,
        "Heavy": """
            You are given a text and you need to fix the language (typos, grammar, ...).
            If needed, fix the formatting and, when relevant, ensure the text is inclusive.
            Please also reformulate the text when needed, use better words and make it more clear.
            Output directly the corrected text, without any comment.
            """,
        "Custom": "",
    }

    # Allow for custom prompts also
    system_name = st.radio("System Prompt", list(system_prompts.keys()), horizontal=True)
    if system_name == "Custom":
        system = st.text_area("Custom prompt", value=dedent(system_prompts["Default"]).strip())
    else:
        system = dedent(system_prompts[system_name]).strip()
        st.code(system, language="text")

    lets_gooo = st.button("Fix", type="primary")


@st.cache_resource()
def cache():
    return {}


if lets_gooo:
    corrected = st.write_stream(ai_stream(system, [dict(role="user", content=text)]))
    cache()[text, system] = corrected
    st.rerun()
else:
    corrected = cache().get((text, system))


def common_prefix(str1, str2):
    min_length = min(len(str1), len(str2))
    for i in range(min_length):
        if str1[i] != str2[i]:
            return str1[:i]
    return str1[:min_length]


def common_suffix(str1, str2):
    min_length = min(len(str1), len(str2))
    for i in range(1, min_length + 1):
        if str1[-i] != str2[-i]:
            return str1[-i + 1 :] if i > 1 else ""
    return str1[-min_length:]


def split_words(text: str) -> list[str]:
    # Split on newlines, while keeping them
    parts = re.findall(r"(\n+|[^\n]+)", text.strip())
    # Split on tokens, i.e. space followed by non-space
    parts = [word for part in parts for word in re.findall(r"^\S+|\s+\S+|\s+$", part)]
    # Split on punctuation
    parts = [word for part in parts for word in re.findall(r"[\w\s]+|[^\w\s]+", part)]
    return parts


def fmt_diff_toggles(diff: list[str], start_with_old_selected: bool = False) -> str:

    # Diff always outputs "- old" then "+ new" word, but both can be empty
    parts: list[str | tuple[str, str]] = []

    for word in diff:
        kind = word[0]
        word = word[2:]

        if kind == " ":
            if parts and isinstance(parts[-1], str):
                parts[-1] += word
            else:
                parts.append(word)
        elif kind == "?":
            continue
        elif kind == "-":
            if parts and isinstance(parts[-1], tuple):
                parts[-1] = (parts[-1][0] + word, parts[-1][1])
            else:
                parts.append((word, ""))
        elif kind == "+":
            if parts and isinstance(parts[-1], tuple):
                parts[-1] = (parts[-1][0], parts[-1][1] + word)
            else:
                parts.append(("", word))
        else:
            raise ValueError(f"Unknown kind: {kind}")

    # Simplify the diff by cleaning (old, new) that start or end with a common substring
    new_parts = []
    for part in parts:
        if isinstance(part, tuple):
            old, new = part
            prefix = common_prefix(old, new)
            suffix = common_suffix(old, new)
            old = old[len(prefix) : -len(suffix) if suffix else None]
            new = new[len(prefix) : -len(suffix) if suffix else None]
            if prefix:
                new_parts.append(prefix)
            new_parts.append((old, new))
            if suffix:
                new_parts.append(suffix)
        else:
            new_parts.append(part)
    parts = new_parts

    style = (
        dedent(
            """
    <style>
        .swaper:checked + label INITIAL_SELECTED {
            user-select: none;
            color: gray;
            border-style: dashed;
            border-color: gray;
        }
        .swaper:not(:checked) + label INITIAL_NOT_SELECTED {
            user-select: none;
            color: gray;
            border-style: dashed;
            border-color: gray;
        }
        .swapable {
            padding: 2px;
            white-space: pre;
            border: 1px solid;
        }
        .swapable:first-child {
            border-top-left-radius: 5px;
            border-bottom-left-radius: 5px;
        }
        .swapable:last-child {
            border-top-right-radius: 5px;
            border-bottom-right-radius: 5px;
        }
        .original {
            border-color: red;
            background-color: rgba(255, 0, 0, 0.05);
            text-decoration-color: red;
        }
        .new {
            border-color: green;
            background-color: rgba(0, 255, 0, 0.05);
            text-decoration-color: green;
        }

        .swapable-label {
            display: inline;
        }
        .whitespace-hint {
            user-select: none;
            color: gray;
        }
        .diff {
            white-space: pre-wrap;
        }
    </style>"""
        )
        .replace("INITIAL_SELECTED", ".original" if start_with_old_selected else ".new")
        .replace("INITIAL_NOT_SELECTED", ".new" if start_with_old_selected else ".original")
        .strip()
    )

    template = """<input type="checkbox" style="display: none;" class="swaper" id={id}>\
<label for={id} class="swapable-label">\
{spans}\
</label>"""
    span_orignal = '<span class="swapable original">{content}</span>'
    span_new = '<span class="swapable new">{content}</span>'

    def fmt_part(part: str) -> str:
        if not part:
            return f'<span class="whitespace-hint">∅</span>'
        else:
            return part.replace("\n", '<span class="whitespace-hint">↵</span><br>')

    colored = ""
    for i, part in enumerate(parts):
        if isinstance(part, tuple):
            spans = []
            if part[0]:
                spans.append(span_orignal.format(content=escape(part[0])))
            if part[1]:
                spans.append(span_new.format(content=escape(part[1])))
            spans = "".join(spans)
            colored += template.format(id=i, spans=spans)
        else:
            colored += f"<span>{part}</span>"
            # colored += f"<span>{part.replace('\n', '<br>')}</span>"

    return f'<p class="diff">{style}{colored}</p>'


if corrected is not None:
    # Compute the difference between the two texts
    words1 = split_words(text)
    words2 = split_words(corrected)

    diff = list(difflib.ndiff(words1, words2))

    st.header("Corrected text")
    options = [":red[Original text]", ":green[New suggestions]"]
    selected = st.radio("Select all", options, index=1, horizontal=True)

    with st.container(border=True):
        st.html(fmt_diff_toggles(diff, start_with_old_selected=selected == options[0]))

    st.warning("This text was written by a generative AI model. You **ALWAYS** need to review it.")
