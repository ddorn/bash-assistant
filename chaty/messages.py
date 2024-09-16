import base64
from dataclasses import dataclass
import dataclasses
from io import BytesIO
import json
import PIL.Image
from openai.types.chat.chat_completion_message import ChatCompletionMessage
import itertools


@dataclass
class MessagePart:

    def to_openai(self):
        raise NotImplementedError()

    def to_anthropic(self):
        raise NotImplementedError()

    # Serialization

    def to_dict(self) -> dict:
        return {
            "class_name": self.__class__.__name__,
            **dataclasses.asdict(self),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MessagePart":
        class_name = data["class_name"]
        for subclass in cls.all_subclasses():
            # Doesn't work for sub-subclasses!
            if subclass.__name__ == class_name:
                return subclass(**{k: v for k, v in data.items() if k != "class_name"})

        raise ValueError(f"Unknown class name: {class_name}")

    @classmethod
    def all_subclasses(cls):
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in c.all_subclasses()]
        )


@dataclass
class TextMessage(MessagePart):
    text: str
    is_user: bool

    def to_openai(self):
        return {
            "role": "user" if self.is_user else "assistant",
            "content": [{"type": "text", "text": self.text}],
        }

    to_anthropic = to_openai


@dataclass
class ImageMessage(MessagePart):
    base_64: str

    @classmethod
    def from_path(cls, path: str):
        img = PIL.Image.open(path)
        # Convert the image to PNG format
        buffered = BytesIO()
        img.save(buffered, format="PNG")

        # Encode the image to base64
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return cls(img_base64)

    def to_openai(self):
        return {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + self.base_64,
                    },
                }
            ],
        }

    def to_anthropic(self):
        return {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": self.base_64,
                    },
                }
            ],
        }


@dataclass
class ToolRequestMessage(MessagePart):
    name: str
    parameters: dict
    id: str

    def to_openai(self):
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": self.id,
                    "type": "function",
                    "function": {"name": self.name, "arguments": json.dumps(self.parameters)},
                }
            ],
        }

    def to_anthropic(self):
        return {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "name": self.name,
                    "input": self.parameters,
                    "id": self.id,
                }
            ],
        }


@dataclass
class ToolOutputMessage(MessagePart):
    id: str
    name: str
    content: str
    canceled: bool = False

    def to_openai(self):
        return {
            "role": "tool",
            # "name": self.name,
            "content": self.content,
            "tool_call_id": self.id,
        }

    def to_anthropic(self):
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": self.id,
                    "content": self.content,
                }
            ],
        }


def merge[T: (dict, list)](a: T, b: T) -> T:
    if isinstance(a, list) and isinstance(b, list):
        return a + b
    elif isinstance(a, dict) and isinstance(b, dict):
        new = {}
        for key in a.keys():
            new[key] = a[key]
        for key in b.keys():
            if key in new and isinstance(new[key], (dict, list)):
                new[key] = merge(new[key], b[key])
            elif key in new:
                assert new[key] == b[key]
            else:
                new[key] = b[key]
        return new
    else:
        raise ValueError(f"Cannot merge {a} and {b}")


class MessageHistory(list[MessagePart]):

    def to_openai(self) -> list[ChatCompletionMessage]:
        formated = []
        # For openai, we need to merge:
        # - an optional assistant TextMessage and the consecutive ToolRequestMessages into a single one
        # - a user TextMessage and subsequent ImageMessages from the same user into a single one

        i = 0
        while i < len(self):
            message = self[i]

            # Merge consecutive user text message and Image messages
            if isinstance(message, TextMessage) and message.is_user:
                new = message.to_openai()
                i += 1
                for attached_image in itertools.takewhile(
                    lambda x: isinstance(x, ImageMessage), self[i:]
                ):
                    new = merge(new, attached_image.to_openai())
                    i += 1

            # Merge an assistant message with subsequent tool requests
            elif isinstance(message, TextMessage) and not message.is_user:
                new = message.to_openai()
                i += 1
                for tool_request in itertools.takewhile(
                    lambda x: isinstance(x, ToolRequestMessage), self[i:]
                ):
                    new = merge(new, tool_request.to_openai())
                    i += 1

            # Just add the message
            else:
                new = message.to_openai()
                i += 1

            formated.append(new)

        return formated

    def to_anthropic(self) -> list[dict]:
        formated = []

        # For anthropic, we need to merge:
        # - all user messages (text and image) and tool outputs into a single message
        # - all other, ie: all assistant messages and tool requests into a single message

        def is_user_message(x):
            return (
                isinstance(x, TextMessage)
                and x.is_user
                or isinstance(x, ImageMessage)
                or isinstance(x, ToolOutputMessage)
            )

        i = 0
        while i < len(self):
            message = self[i]

            # Merge consecutive user text/image message and tool outputs
            if is_user_message(message):
                new = message.to_anthropic()
                i += 1
                for part in itertools.takewhile(is_user_message, self[i:]):
                    new = merge(new, part.to_anthropic())
                    i += 1
            else:
                new = message.to_anthropic()
                i += 1
                for part in itertools.takewhile(lambda x: not is_user_message(x), self[i:]):
                    new = merge(new, part.to_anthropic())
                    i += 1

            formated.append(new)

        return formated

    def to_dict(self) -> list[dict]:
        return [message.to_dict() for message in self]

    @classmethod
    def from_dict(cls, data: list[dict]) -> "MessageHistory":
        return cls(MessagePart.from_dict(message) for message in data)
