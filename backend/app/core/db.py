from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
import chromadb

from app.core.config import settings

db = None


def init_db():
    embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
    chroma_client = chromadb.HttpClient(
        host=settings.CHROMA_HOST, port=settings.CHROMA_PORT
    )
    global db
    db = Chroma(
        collection_name=settings.CHROMA_COLLECTION,
        embedding_function=embeddings,
        client=chroma_client,
    )
    print("DB initialized")


def get_db():
    global db
    return db


def get_relevant_documents(question: str) -> List[Document]:
    db = get_db()
    retriever = db.as_retriever(
        search_type="mmr", search_kwargs={"k": 5, "fetch_k": 50}
    )
    relevant_docs = retriever.invoke(question)
    return relevant_docs
