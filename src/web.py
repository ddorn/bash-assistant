import streamlit as st


PAGES = [
    "fix",
    "bai_web",
    "battery",
    # "dashboard",
]

page = st.navigation([st.Page(f"pages/{name}.py") for name in PAGES])

page.run()
