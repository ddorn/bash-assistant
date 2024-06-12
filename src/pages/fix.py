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
    If needed, fix also the formatting and ensure the text is gender neutral.
    Output directly the corrected text, without any comment.
    """.strip(),
        "Heavy": """
    You are given a text and you need to fix the language (typos, grammar, ...).
    If needed, fix also the formatting and ensure the text is gender neutral.
    "Please reformulate the text when needed, use better words and make it more clear when possible.",
    Output directly the corrected text, without any comment.
    """.strip(),
        "Custom": "",
    }

    # Allow for custom prompts also
    system_name = st.selectbox("System Prompt", list(system_prompts.keys()))
    if system_name == "Custom":
        system = st.text_area("Custom prompt", value=system_prompts["Default"])
    else:
        system = system_prompts[system_name]
        st.markdown(system)

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


def fmt_diff_html(diff: list[str], select_new: bool) -> str:
    no_select = "user-select: none; color: gray;"

    colored = ""
    for word in diff:
        kind = escape(word[0])
        word = word[2:]

        if kind == " ":
            colored += f"<span>{word}</span>"
            continue
        elif kind == "?":
            continue

        colored += "<span style='"

        if kind == "-":
            if select_new:
                colored += no_select
            else:
                colored += "color: red;"
        elif kind == "+":
            if not select_new:
                colored += no_select
            else:
                colored += "color: green;"
        else:
            continue
        colored += "'>" + word + "</span>"

    return colored.replace("\n", "<br>")


def fmt_diff_toggles(diff: list[str]) -> str:
    start = dedent(
        """
    <style>
        .swaper:checked + label .new {
            user-select: none;
            color: gray;
            border: 1px dashed;
        }
        .swaper:not(:checked) + label .original {
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
    </style>"""
    )

    template = """
<input type="checkbox" style="display: none;" class="swaper" id={id}>
<label for={id} class="swapable-label">
    <span class="swapable original">{content1}</span><span class="swapable new">{content2}</span>
</label>"""

    # Diff always outputs "- old" then "+ new" word, but both can be empty
    parts: list[str | tuple[str, str]] = []

    for word in diff:
        kind = word[0]
        word = escape(word[2:])

        if kind == " ":
            if parts and isinstance(parts[-1], str):
                parts[-1] += word
            else:
                parts.append(word)
        elif kind == "?":
            continue
        elif kind == "-":
            # if parts and isinstance(parts[-1], tuple):
            #     if parts[-1][1] == "":
            #         parts[-1] = (parts[-1][0] + word, "")
            #     else:
            #         parts.append((word, ""))
            # else:
            parts.append((word, ""))
        elif kind == "+":
            # if parts and isinstance(parts[-1], tuple):
            #     parts[-1] = (parts[-1][0], word)
            # else:
            if parts and isinstance(parts[-1], tuple) and parts[-1][1] == "":
                parts[-1] = (parts[-1][0], word)
            else:
                parts.append(("", word))
        else:
            raise ValueError(f"Unknown kind: {kind}")

    # Escape the text

    colored = start
    for i, part in enumerate(parts):
        if isinstance(part, tuple):
            colored += template.format(id=i, content1=part[0], content2=part[1])
        else:
            colored += f"<span>{part}</span>"

    return f"<p>{colored}</p>"


def split_words(text: str) -> list[str]:
    # Also split on newlines
    parts = re.findall(r"(\n+|[^\n]+)", text.strip())
    words = [word for part in parts for word in re.findall(r"\S+|\s+", part)]
    return words


if corrected is not None:
    st.write(
        "*:green[Green text] is the corrected version, :red[red text] is the original version.*"
    )

    # Compute the difference between the two texts
    words1 = split_words(text)
    words2 = split_words(corrected)

    diff = list(difflib.ndiff(words1, words2))

    st.header("Word level toggles")
    st.html(fmt_diff_toggles(diff))

    st.header("Global toggle")
    select_old = st.checkbox("Make old text selectable", value=False)
    colored = fmt_diff_html(diff, select_new=not select_old)
    st.html(colored)

    st.header("Plain diff")
    st.code("\n".join(diff), language="diff")
