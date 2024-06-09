from pydantic import BaseModel
from typing import List


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str
    sources: List[str] | None = None
