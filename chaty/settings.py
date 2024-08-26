from dataclasses import dataclass
import chainlit as cl

from messages import MessageHistory
from llms import MODELS, LLM


@dataclass
class Settings:
    model: str
    tool_use: bool
    history: MessageHistory
    settings_message: cl.Message | None = None

    @classmethod
    def from_user_session(cls) -> "Settings":
        settings = cl.user_session.get("settings")
        if settings is None:
            settings = cls(
                model=MODELS[0].nice_name,
                tool_use=True,
                history=MessageHistory(),
                settings_message=None,
            )
            cl.user_session.set("settings", settings)

        return settings

    get = from_user_session

    @property
    def current_model(self) -> LLM:
        return next(m for m in MODELS if m.nice_name == self.model)

    def get_next_model(self) -> str:
        model_idx = next(i for i, m in enumerate(MODELS) if m.nice_name == self.model)
        next_model = MODELS[(model_idx + 1) % len(MODELS)]
        return next_model.nice_name

    def make_settings_widget_data(self) -> tuple[str, list[cl.Action]]:
        """
        Return the text and action button data for the settings message.
        """
        next_model = self.get_next_model()

        actions = [
            cl.Action(name="switch-model", value=next_model, label=f"Switch to {next_model}"),
            cl.Action(name="toggle-tool-use", value="", label="Toggle tool use"),
        ]

        settings = f"""
- **Model**: {self.model}
- **Tool Use**: {'❌✅'[self.tool_use]}
""".strip()

        return settings, actions

    async def update_settings_message(self):
        """Send or update the settings message and its buttons."""
        settings, actions = self.make_settings_widget_data()

        if self.settings_message is None:
            settings_message = cl.Message(settings, actions=actions)
            self.settings_message = settings_message
            await settings_message.send()
        else:
            settings, actions = self.make_settings_widget_data()
            self.settings_message.content = settings
            for action in self.settings_message.actions:
                await action.remove()
            self.settings_message.actions = actions
            await self.settings_message.update()


@cl.action_callback("switch-model")
async def switch_model_action(action):
    model = action.value
    settings = Settings.from_user_session()
    settings.model = model

    await settings.update_settings_message()


@cl.action_callback("toggle-tool-use")
async def toggle_tool_use_action(action):
    settings = Settings.from_user_session()
    settings.tool_use = not settings.tool_use

    await settings.update_settings_message()
