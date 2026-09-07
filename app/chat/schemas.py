from pydantic import BaseModel

class ChatRequest(BaseModel):
    model: str = "llama3"
    question: str
    context: str = ""
    temperature: float = 0.2
    max_tokens: int = 512
