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


def fmt_diff_toggles(diff: list[str], start_with_old_selected: bool = False) -> str:
    style = (
        dedent(
            """
    <style>
        .swaper:checked + label INITIAL_SELECTED {
            user-select: none;
            color: gray;
            border: 1px dashed;
        }
        .swaper:not(:checked) + label INITIAL_NOT_SELECTED {
            user-select: none;
            color: gray;
            border: 1px dashed;
        }
        .swapable {
            border: 1px solid;
            padding: 2px;
            white-space: pre;
        }
        .original {
            border-color: red;
            background-color: rgba(255, 0, 0, 0.05);
            border-radius: 5px 0 0 5px;
            text-decoration-color: red;
        }
        .new {
            border-color: green;
            background-color: rgba(0, 255, 0, 0.05);
            border-radius: 0 5px 5px 0;
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
<span class="swapable original">{content1}</span><span class="swapable new">{content2}</span>\
</label>"""

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
                if parts[-1][1] == "":
                    parts[-1] = (parts[-1][0] + word, "")
                else:
                    parts.append((word, ""))
            else:
                parts.append((word, ""))
        elif kind == "+":
            if parts and isinstance(parts[-1], tuple):
                parts[-1] = (parts[-1][0], word)
            else:
                parts.append(("", word))
        else:
            raise ValueError(f"Unknown kind: {kind}")

    # Escape the text

    def fmt_part(part: str) -> str:
        if not part:
            return f'<span class="whitespace-hint">∅</span>'
        else:
            return part.replace("\n", '<span class="whitespace-hint">↵</span><br>')

    colored = ""
    for i, part in enumerate(parts):
        if isinstance(part, tuple):
            colored += template.format(id=i, content1=fmt_part(part[0]), content2=fmt_part(part[1]))
        else:
            colored += f"<span>{part}</span>"
            # colored += f"<span>{part.replace('\n', '<br>')}</span>"

    return f'<p class="diff">{style}{colored}</p>'


def split_words(text: str) -> list[str]:
    # Also split on newlines
    parts = re.findall(r"(\n+|[^\n]+)", text.strip())
    words = [word for part in parts for word in re.findall(r"\S+|\s+", part)]
    return words


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

    # st.write("*This text was generated by an AI model. You **ALWAYS** need to review it.*")
    # Nicer
    st.warning("This text was written by a generative AI model. You **ALWAYS** need to review it.")
