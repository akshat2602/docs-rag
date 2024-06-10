from fastapi import APIRouter
from typing import Any, List
from pydantic import BaseModel

from app.core.openai import answer_question

api_router = APIRouter()


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str
    sources: List[str] | None = None


@api_router.post("/answer", response_model=QuestionResponse)
def answer_question_api(request: QuestionRequest) -> Any:
    """
    Answer a question using RAG.
    """
    answer, relevant_docs = answer_question(request.question)
    sources = []
    for relevant_doc in relevant_docs:
        sources.append(relevant_doc.metadata["path"])
    print(relevant_docs)
    return {"answer": answer, "sources": sources}
