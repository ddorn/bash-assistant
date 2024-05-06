import os
import streamlit as st


def unneeded_packages():
    unneeded = os.popen('pacman -Qdtq').read().strip().split('\n')
    if unneeded:
        unneeded = "\n- ".join(unneeded)
        st.write(f"""
Some packages were installed as dependencies but are no longer needed (`pacman -Qdtq`):
- {unneeded}

You can remove them with
```bash
sudo pacman -Rns $(pacman -Qdtq)
```
""")





def dashboard():
    st.title("Hey Diego! What's important for you right now?")

    unneeded_packages()


if __name__ == "__main__":
    dashboard()
