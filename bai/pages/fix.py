import difflib
import re
import streamlit as st

from utils import ai_stream


text = st.text_area("Text to fix")
heavy = st.checkbox("Also reformulate")


def fix(text, heavy):
    system = """
You are given a text and you need to fix the language (typos, grammar, ...).
If needed, fix also the formatting and ensure the text is gender neutral. HEAVY
Output directly the corrected text, without any comment.
""".strip()

    if heavy:
        system = system.replace(
            "HEAVY",
            "Please reformulate the text when needed, use better words and make it more clear when possible.",
        )
    else:
        system = system.replace(" HEAVY", "")

    result = st.write_stream(ai_stream(system, [dict(role="user", content=text)]))
    return result


@st.cache_resource()
def cache():
    return {}


def fmt_diff_html(diff: list[str], select_new: bool) -> str:
    no_select = "user-select: none; color: gray;"

    colored = ""
    for word in diff:
        kind = word[0]
        word = word[2:]

        if kind == " ":
            colored += word
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

    return colored


if st.button("Fix"):
    corrected = fix(text, heavy)
    cache()[text, heavy] = corrected
    st.rerun()
else:
    corrected = cache().get((text, heavy))

if corrected is not None:
    # Compute the difference between the two texts
    words1 = re.findall(r"(\w+|\W+)", text.strip())
    words2 = re.findall(r"(\w+|\W+)", corrected.strip())

    diff = list(difflib.ndiff(words1, words2))
    # past, corrected = utilfmt_diff(diff)

    st.header("Corrected text")
    colored = fmt_diff_html(diff, select_new=True)
    st.write(colored, unsafe_allow_html=True)

    st.header("Old text")
    colored = fmt_diff_html(diff, select_new=False)
    st.write(colored, unsafe_allow_html=True)
