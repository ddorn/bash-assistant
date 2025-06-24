import streamlit as st
from pathlib import Path


PAGES = [
    "fix",
    "bai_web",
    "battery",
    # "dashboard",
]
PAGES = [p.stem for p in Path("src/pages/").glob("*.py")]

page = st.navigation([st.Page(f"pages/{name}.py") for name in PAGES])

page.run()
