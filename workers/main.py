from hatchet_sdk import Hatchet
from dotenv import load_dotenv
from langchain_community.document_loaders import GithubFileLoader
import os

load_dotenv()

hatchet = Hatchet()


@hatchet.workflow(on_events=["rag:crawl"])
class RAGCrawlerWorkflow:
    @hatchet.step()
    def crawl(self, context):
        # Collect all markdown urls and then crawl to get the data
        loader = GithubFileLoader(
            repo="hatchet-dev/hatchet",
            access_token=os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"],
            github_api_url="https://api.github.com",
            file_filter=lambda file_path: file_path.endswith(
                (".md", ".mdx")
            ) and "frontend/docs" in file_path,
        )
        file_paths = loader.get_file_paths()
        return {
            "status": "crawl done",
            "file_paths": file_paths,
        }

    @hatchet.step(parents=["crawl"])
    def push_embed(self, context):
        file_paths = context.step_output("crawl")["file_paths"]
        for file in file_paths:
            context.log(f"Pushing embeddings for {file.get('path')}")
            hatchet.client.event.push("rag:embeddings", {"file": file})
        return {
            "status": "pushed embeddings"
        }


@hatchet.workflow(on_events=["rag:embeddings"])
class RAGEmbeddingsWorkflow:
    @hatchet.step()
    def start(self, context):
        # Collect markdown data and then clean and then embed
        print("executed start")
        pass

    @hatchet.step(parents=["start"])
    def embeddings(self, context):
        print("executed embeddings")
        pass

    @hatchet.step(parents=["embeddings"])
    def store(self, context):
        print("executed store")
        pass


worker = hatchet.worker("docs-rag-worker")
worker.register_workflow(RAGCrawlerWorkflow())
worker.register_workflow(RAGEmbeddingsWorkflow())


worker.start()
