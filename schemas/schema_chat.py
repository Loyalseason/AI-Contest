from pydantic import BaseModel
from typing_extensions import List

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    model: str
