from concurrent.futures import ThreadPoolExecutor
from math import ceil
from textwrap import indent
from pydantic import BaseModel
import streamlit as st

import models
from llms import ai_structured


class BestOfN(BaseModel):
    reasoning: str
    first_4_words_of_best_option: str
    best_option_0_based_index: int


@st.cache_data
def best_of_n(paragraphs: list[str]) -> int:
    out = ai_structured(
        f"Which of the following {len(paragraphs)} paragraphs is most interesting for a reader? First think about why each is or isn't interesting.",
        "\n\n".join(f"{i}. {p}" for i, p in enumerate(paragraphs)),
        BestOfN,
        model=models.TREE_BOOK,
    )
    return out.best_option_0_based_index


@st.cache_data
def best_of_n_batched(paragraphs: list[list[str]]) -> list[int]:
    def call_one(chunk: list[str]) -> BestOfN:
        return ai_structured(
            f"Which of the following {len(chunk)} paragraphs is most interesting for a reader? First think about why each is or isn't interesting.",
            "\n\n".join(f"{i}. {p}" for i, p in enumerate(chunk)),
            BestOfN,
            model=models.TREE_BOOK,
        )

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(call_one, paragraphs))
    return [r.best_option_0_based_index for r in results]


st.write(best_of_n_batched([["a", "b", "c"], ["d", "e", "f"]]))


def batched_nice(iterable, n):
    """Batch an iterable into chunks of size at most n, maximizing the number of chunks and their size."""

    nb_chunks = ceil(len(iterable) / n)
    chunk_size = len(iterable) / nb_chunks

    chunks = []
    for i in range(nb_chunks):
        start = int(i * chunk_size)
        end = int((i + 1) * chunk_size)
        chunks.append(iterable[start:end])

    return chunks


class Node(BaseModel):
    content: str
    children: list["Node"] = []

    def to_markdown(self) -> str:
        if not self.children:
            return "- " + self.content
        return f"- {self.content}\n{indent('\n'.join(child.to_markdown() for child in self.children), '  ')}"

    def to_text(self):
        if not self.children:
            return self.content
        return (
            self.content
            + "\n"
            + "\n".join(indent(child.to_text(), "\t") for child in self.children)
        )

    def to_opml(self, root=False) -> str:
        if not self.children:
            child = ""
        else:
            child = "\n".join(child.to_opml() for child in self.children)
            child = indent(child, "  ")
            child = f"\n{child}\n"
        outline = f"<outline text='{self.content}'>{child}</outline>"
        if root:
            return f'<opml version="1.0">\n<body>\n{indent(outline, "  ")}\n</body>\n</opml>'
        else:
            return outline


def paragraph_tree(paragraphs: list[str], branching_factor: int = 4) -> Node:
    # We build a tree bottom-up, starting with all the paragraphs as leaves
    # and the parent nodes as the best of the children.

    this_level = [Node(content=para) for para in paragraphs]
    while len(this_level) > 1:
        next_level = []
        for chunk in batched_nice(this_level, branching_factor):
            best_idx = best_of_n([node.content for node in chunk])
            parent = Node(content=chunk[best_idx].content, children=chunk)
            next_level.append(parent)
        this_level = next_level

    return this_level[0]


text = st.text_area("Book to make recursive")

split_by = st.radio("Split by", ["separator", "length"], horizontal=True)
if split_by == "separator":
    separator = st.text_input("Separator", r"\n\n")
    separator = separator.encode().decode("unicode_escape")
    paragraphs = text.split(separator)
elif split_by == "length":
    length = st.number_input("Length", 800)
    paragraphs = [text[i : i + length] for i in range(0, len(text), length)]
else:
    raise ValueError(split_by)

st.write("Number of paragraphs:", len(paragraphs))
with st.expander("Show paragraphs"):
    st.write(paragraphs)


if not st.button("Go"):
    st.stop()


tree = paragraph_tree(paragraphs)

st.code(tree.to_text(), language="markdown")
# st.code(tree.to_opml(root=True), language="markdown")
