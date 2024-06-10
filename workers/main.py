from hatchet_sdk import Hatchet
from dotenv import load_dotenv
from langchain_community.document_loaders import GithubFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import chromadb

load_dotenv()

hatchet = Hatchet()

embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
chroma_client = chromadb.HttpClient(
    host=os.environ["CHROMA_HOST"], port=os.environ["CHROMA_PORT"]
)
db = Chroma(
    collection_name=os.environ["CHROMA_COLLECTION"],
    embedding_function=embeddings,
    client=chroma_client,
)


@hatchet.workflow(on_events=["rag:crawl"])
class RAGCrawlerWorkflow:
    @hatchet.step()
    def crawl(self, context):
        # Collect all markdown urls and then crawl to get the data
        loader = GithubFileLoader(
            repo="hatchet-dev/hatchet",
            access_token=os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"],
            github_api_url="https://api.github.com",
            file_filter=lambda file_path: file_path.endswith((".md", ".mdx"))
            and "frontend/docs" in file_path,
        )
        file_paths = loader.get_file_paths()
        return {
            "status": "crawl done",
            "file_paths": file_paths,
        }

    @hatchet.step(parents=["crawl"])
    def push_for_embed(self, context):
        file_paths = context.step_output("crawl")["file_paths"]
        for file in file_paths:
            context.log(f"Pushing embeddings for {file.get('path')}")
            hatchet.client.event.push("rag:embeddings", {"file": file})
        return {"status": "pushed embeddings"}


@hatchet.workflow(on_events=["rag:embeddings"])
class RAGEmbeddingsWorkflow:
    @hatchet.step(timeout="5m")
    def fetch_document(self, context):
        file = context.workflow_input()["file"]
        context.log(f"Downloading {file.get('path')}")
        loader = GithubFileLoader(
            repo="hatchet-dev/hatchet",
            access_token=os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"],
            github_api_url="https://api.github.com",
            file_filter=lambda file_path: file["path"] in file_path,
        )
        document = loader.load()[0].page_content
        return {
            "status": "loaded document",
            "document": document,
        }

    @hatchet.step(parents=["fetch_document"], timeout="10m")
    def store_embeddings(self, context):
        document = context.step_output("fetch_document")["document"]
        workflow_input = context.workflow_input()
        context.log(f"Embedding {workflow_input['file'].get('path')}")
        doc = Document(
            page_content=document,
            metadata=workflow_input['file'],
        )
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
        processed_document = text_splitter.split_documents([doc])
        for doc in processed_document:
            context.log(f"Adding text to db: {doc.metadata}")
        db.add_texts(
            texts=[doc.page_content for doc in processed_document],
            metadatas=[doc.metadata for doc in processed_document],
        )
        context.log(f"Stored embeddings for {workflow_input['file'].get('path')}")
        return {"status": "embedded"}


worker = hatchet.worker("docs-rag-worker")
worker.register_workflow(RAGCrawlerWorkflow())
worker.register_workflow(RAGEmbeddingsWorkflow())


worker.start()
