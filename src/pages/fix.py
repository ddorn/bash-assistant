from html import escape
import difflib
import re
from textwrap import dedent
import streamlit as st
from pprint import pprint

from utils import ai_stream


with st.container(border=True):
    text = st.text_area("Text to fix")

    system_prompts = {
        "Default": """
    You are given a text and you need to fix the language (typos, grammar, ...). Keep the newline where they are, keep the same paragraph structure even if it seems incorrect. Output directly the corrected text, without any comment. 
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
    #text = text.replace("\n\n", "\n@@@\n") #we use ~*~ as paragraph separator
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
</label>""" # the bad indentation is indended to remove unwanted whitespace when copying the text

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
            if parts and isinstance(parts[-1], tuple):
                if parts[-1][1] == "":
                    parts[-1] = (parts[-1][0] + word, "")
                else:
                    parts.append((word, ""))
            else:
                parts.append((word, ""))
        elif kind == "+":
            if parts and isinstance(parts[-1], tuple):
                parts[-1] = (parts[-1][0], parts[-1][1] + word)
            else:
                parts.append(("", word))
        else:
            raise ValueError(f"Unknown kind: {kind}")

    # Escape the text

    def fmt_part(part: str) -> str:
        if not part:
            return f'<span style="user-select: none">∅</span>'
        else:
            return part.replace("\n", '<span style="user-select: none">↵</span><br>') #new line handling

    colored = start

    for i, part in enumerate(parts):
        if isinstance(part, tuple):
            colored += template.format(id=i, content1=fmt_part(part[0]), content2=fmt_part(part[1]))
        else:
            colored += f"<span>{fmt_part(part)}</span>"

    return f"<p>{colored.replace("\n", "")}</p>"


def split_words(text: str) -> list[str]:
    # Split text into words and whitespace (new function that has granular split to avoid block diff)
    return re.findall(r'\w+|[^\w\s]+|\s+', text)


def split_into_blocks(text: str) -> list[str]:
    # Splits the text into blocks by paragraphs or double newlines
    return re.split(r'\n{2}', text.strip())

def diff_text_blocks(text1: str, text2: str) -> list[str]:
    blocks1 = split_into_blocks(text1)
    blocks2 = split_into_blocks(text2)
    all_diffs = []
    
    # Ensure both lists have the same number of blocks by padding the shorter one with empty strings
    max_len = max(len(blocks1), len(blocks2))
    blocks1.extend([''] * (max_len - len(blocks1)))
    blocks2.extend([''] * (max_len - len(blocks2)))

    for block1, block2 in zip(blocks1, blocks2):
        words1 = split_words(block1)
        words2 = split_words(block2)
        block_diff = list(difflib.ndiff(words1, words2))
        all_diffs.extend(block_diff + ['  \n'] + ['  \n'],)  # Add a newline to separate blocks

    return all_diffs

if corrected is not None:
    st.write(
        "*:green[Green text] is the corrected version, :red[red text] is the original version. You might need to copy without the style to discard unnecessary text.*"
    )
    

    # Compute the difference between the two texts by blocks
    diff = diff_text_blocks(text, corrected)

    st.header("Word level toggles")
    st.html(fmt_diff_toggles(diff))

    st.header("Global toggle")
    select_old = st.checkbox("Make old text selectable", value=False)
    colored = fmt_diff_html(diff, select_new=not select_old)
    st.html(colored)
