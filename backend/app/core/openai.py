from typing import List
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

from app.core.config import settings
from app.core.db import get_relevant_documents

llm = None


def init_openai():
    global llm
    llm = OpenAI(api_key=settings.OPENAI_API_KEY)


def get_llm():
    global llm
    return llm


def answer_question(question: str) -> tuple[str, List[Document]]:
    """
    Answer a question using RAG.
    First step: Fetch relevant documents from the db
    Second step: Pass the documents and the question to ChatGPT
    Third step: Return the answer
    """
    relevant_docs = get_relevant_documents(question)
    prepped_prompt = get_magic_prompt()
    prompt = PromptTemplate.from_template(template=prepped_prompt)
    runnable_prompt = prompt.invoke(
        {"relevant_docs": relevant_docs, "question": question}
    )
    llm = get_llm()
    answer = llm.invoke(runnable_prompt)
    return (answer, relevant_docs)


def get_magic_prompt() -> str:
    return """
        Here is a list of documents:
        {relevant_docs}
        List of documents ends here.
        Here's a question about the documents:
        {question}
        Generate an answer for the question using only the relevant documents,
        do not make things up.
        If you can't find an answer, say so.
        Also, try to give code examples if possible.
        Answer the question in markdown format.:
        """
