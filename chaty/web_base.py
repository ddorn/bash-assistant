import asyncio
from enum import StrEnum
import json
from typing import Callable
import streamlit as st
from messages import (
    MessageHistory,
    TextMessage,
    MessagePart,
    ImageMessage,
    ToolRequestMessage,
    ToolOutputMessage,
)
from llms import OpenAILLM
from tools import Tool, create_tool_from_function
from streamlit_pills import pills as st_pills


class WebChat:
    def __init__(self):
        self.all_tools: list[Tool] = []
        self.messages: MessageHistory = st.session_state.setdefault("messages", MessageHistory())
        self.unprocessed_messages: MessageHistory = st.session_state.setdefault(
            "unprocessed_messages", MessageHistory()
        )
        self.model = OpenAILLM("GPT 4o Mini", "gpt-4o-mini")
        self.tool_requests_containers: dict = {}  # {part.id: st.container}

        # While developing, and the script reloads, the class of the messages
        # get out of sync, as the class is redefined/reimported. This is a
        # workaround to fix that.
        classes = [TextMessage, ImageMessage, ToolRequestMessage, ToolOutputMessage]
        for message in self.messages:
            message.__class__ = next(
                c for c in classes if c.__name__ == message.to_dict()["class_name"]
            )
        for message in self.unprocessed_messages:
            message.__class__ = next(
                c for c in classes if c.__name__ == message.to_dict()["class_name"]
            )

    def tool[T: Callable](self, tool: T) -> T:
        """Decorator to register a tool in the chat."""
        self.all_tools.append(create_tool_from_function(tool))
        return tool

    async def main(self):
        st.title("Diego's AI Chat")
        st.button("Clear chat", on_click=lambda: st.session_state.pop("messages"))

        with st.expander("Tools"):
            for tool in self.all_tools:
                st.write(f"### {tool.name}")
                st.write(tool.description)
                st.write(f"Parameters: {tool.parameters}")
                st.write(f"Required: {tool.required}")
                st.code(json.dumps(tool.as_json(), indent=2), language="json")

        with st.expander("Message history as JSON"):
            st.code(json.dumps(self.messages.to_openai(), indent=2), language="json")

        st.markdown(
            """
<style>
    [data-testid=stChatMessage] {
        padding: 0.5rem;
        margin: -0.5rem 0;
    }
</style>
""",
            unsafe_allow_html=True,
        )

        # Display each message in the message history
        self.show_messages(*self.messages)

        for part in self.unprocessed_messages[:]:
            if isinstance(part, ToolRequestMessage):
                output = await self.confirm_and_run(part)
                if output is not None:
                    self.unprocessed_messages.remove(part)
                    self.messages.append(part)
                    self.messages.append(output)
                    st.rerun()
            elif isinstance(part, TextMessage) and part.is_user:
                # Generate a response
                self.show_messages(part)
                new_parts = await self.model(
                    "Be straightforward.", MessageHistory((*self.messages, part)), self.all_tools
                )
                self.unprocessed_messages.extend(new_parts)
                self.messages.append(part)
                self.unprocessed_messages.remove(part)
                st.rerun()

            else:
                self.unprocessed_messages.remove(part)
                self.messages.append(part)
                self.show_messages(part)

        # Get the user's message
        if not self.unprocessed_messages:
            user_message = st.chat_input()
            if user_message:
                new_part = TextMessage(user_message, is_user=True)
                self.unprocessed_messages.append(new_part)
                st.rerun()

    def show_messages(self, *messages: MessagePart):
        for message in messages:
            # We put tool outputs next to its tool.
            if isinstance(message, ToolOutputMessage):
                with self.tool_requests_containers[message.id]:
                    st.write(f"**output**: {message.content}")
                continue

            role = message.to_openai()["role"]
            container = st.chat_message(name=role)
            with container:
                if isinstance(message, TextMessage):
                    st.write(message.text)
                elif isinstance(message, ImageMessage):
                    st.warning("Image messages are not supported yet.")
                elif isinstance(message, ToolRequestMessage):
                    self.tool_requests_containers[message.id] = container
                    s = f"Request to use **{message.name}**\n"
                    for key, value in message.parameters.items():
                        s += f"{key}: {self.to_inline_or_code_block(value)}\n"
                    st.write(s)
                else:
                    st.warning(f"Unsupported message type: {type(message)}")

    def to_inline_or_code_block(self, value):
        if "\n" in str(value):
            return f"\n```\n{value}\n```"
        else:
            return f"`{value}`"

    async def confirm_and_run(self, part: ToolRequestMessage):
        the_tool: Tool = next(t for t in self.all_tools if t.name == part.name)

        self.show_messages(part)
        with self.tool_requests_containers[part.id]:
            action = st_pills(
                None,
                list(ToolActions),
                index=None,
                label_visibility="collapsed",
                key="actions" + part.id,
            )
        if action is None:
            return
        elif action == ToolActions.ALLOW_AND_RUN:
            out = await the_tool.run(**part.parameters)
            canceled = False
        elif action == ToolActions.DENY:
            out = "Tool call denied by user."
            canceled = True
        else:
            raise NotImplementedError(action)

        return ToolOutputMessage(part.id, the_tool.name, out, canceled)

    def run(self):
        asyncio.run(self.main())


class ToolActions(StrEnum):
    ALLOW_AND_RUN = "✅ Allow and Run"
    DENY = "❌ Deny"
    # EDIT = "✏️ Edit"
