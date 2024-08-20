from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import streamlit as st

from src import utils, constants


DATA = utils.DATA / "chats"
DATA.mkdir(exist_ok=True, parents=True)

DEFAULT_NAME = "Untitled"


def approx_text_height(text: str) -> int:
    height = 0
    for line in text.split("\n"):
        height += 1 + len(line) // 50
    return height


@dataclass
class GenerationArgs:
    model: str
    temperature: float


GENERATION_ARGS = GenerationArgs(model=constants.CHEAPEST_MODEL, temperature=0.2)


@dataclass
class Chat:
    messages: list
    path: Path
    name: str

    @classmethod
    def load(cls, path: Path) -> "Chat":
        data = json.loads(path.read_text())
        return cls(**data, path=path)

    def save(self):
        data = self.__dict__.copy()
        del data["path"]
        self.path.write_text(json.dumps(data))

    def edit(self, i: int, new_content: str):
        self.messages[i]["content"] = new_content
        self.save()

    def delete_msg(self, i: int):
        self.messages.pop(i)
        self.save()

    def preprocess_messages(
        self, up_to: int | None = None
    ) -> tuple[str | None, list[dict[str, str]]]:
        system = None
        messages = []
        for message in self.messages[:up_to]:
            if message["role"] == "system":
                assert system is None
                system = message["content"]
            elif message["content"]:
                messages.append(message)
        return system, messages

    def generate(self, regenerate_idx: int | None = None):
        model = GENERATION_ARGS.model
        if model is None:
            content = "new message!"
        else:
            messages = self.preprocess_messages(up_to=regenerate_idx)
            content = st.write_stream(utils.ai_stream(*messages, **GENERATION_ARGS.__dict__))

            if self.name == DEFAULT_NAME:
                self.generate_name()

        if regenerate_idx is not None:
            return self.edit(regenerate_idx, content)

        self.messages.append(dict(role="assistant", content=content))
        self.messages.append(dict(role="user", content=""))
        self.save()

    def generate_name(self):
        system, messages = self.preprocess_messages()
        start = ""
        if system is not None:
            start += system + "\n\n"
        if messages:
            start += messages[0]["content"]

        system = (
            "Generate a 2-4 word title for the conversation starting below. Output only 2-4 words.\n"
            'Good examples include: "Pygame engine name", "Draft paper intro", "Write Exif with python", "Mail Ben talk"'
        )

        self.name = utils.ai_chat(
            system=system,
            messages=[
                dict(role="user", content=start),
                dict(role="system", content=system.replace("below", "above")),
            ],
            model=constants.CHEAPEST_MODEL,
        )
        self.save()

    def show(self, edit_mode: bool = False):

        # Allow to edit the name
        cols = st.columns([1, 12])

        if cols[0].button("♻"):
            self.generate_name()

        if edit_mode:
            new = cols[1].text_input("Name", self.name)
            if new != self.name:
                self.name = new
                self.save()
                st.rerun()
        else:
            cols[1].subheader(self.name)

        if not self.messages:
            self.messages.append(dict(role="system", content=""))
            self.messages.append(dict(role="user", content=""))

        for i, message in enumerate(self.messages):
            avatar = {"system": "⚙", "user": "👤", "assistant": "🤖"}[message["role"]]
            role = message["role"]
            name = avatar + " " + message["role"].capitalize()
            is_user = role == "user"
            is_system = role == "system"

            cols = st.columns([1, 10] if is_user else [10, 1])
            with cols[not is_user]:
                if edit_mode and not is_system and st.button("🗑", f"bin-{i}"):
                    self.delete_msg(i)
                    st.rerun()

                regenerate = role == "assistant" and st.button("♻", key=f"redo-{i}")

            with cols[is_user]:
                if regenerate:
                    self.generate(regenerate_idx=i)
                elif edit_mode or is_user or is_system:
                    height = 15 * max(4, approx_text_height(message["content"]))
                    new = st.text_area(name, message["content"], key=f"content-{i}", height=height)
                    if new != message["content"]:
                        self.edit(i, new)
                else:
                    st.markdown(f"**{name}**  \n" + message["content"])

        # We want the message before the button
        generated_msg = st.empty()

        # Generate a new message
        if st.button("Generate", use_container_width=True):
            with generated_msg:
                self.generate()
                st.rerun()


def new_chat():
    path = DATA / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    chat = Chat(messages=[], path=path, name=DEFAULT_NAME)
    chat.save()


def main():
    st.title("Diego's assistant")

    all_chats = [Chat.load(path) for path in DATA.glob("*.json")]
    # Created time
    all_chats.sort(key=lambda x: x.path.name, reverse=True)

    with st.sidebar:
        GENERATION_ARGS.model = st.selectbox("Model", constants.MODELS + [None])
        GENERATION_ARGS.temperature = st.slider("Temperature", 0.0, 1.5, 0.2)
        edit_mode = st.checkbox("Edit mode")

        st.button("Create new chat", on_click=new_chat)

        # The next line DOESNT WORK. It makes the app need more reloads every time a button is clicked.
        # selected_chat = st.radio("Select chat", all_chats, format_func=lambda x: x.name)

        # To work around this, we selected the chat by index
        idx = st.radio(
            "Select chat", range(len(all_chats)), format_func=lambda x: all_chats[x].name
        )
        selected_chat: Chat = all_chats[idx]

    selected_chat.show(edit_mode=edit_mode)


if __name__ == "__main__":
    main()
