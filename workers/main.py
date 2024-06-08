from hatchet_sdk import Hatchet
from dotenv import load_dotenv

load_dotenv()

hatchet = Hatchet()

@hatchet.workflow(on_events=["rag:crawl"])
class RAGCrawlerWorkflow:
    @hatchet.step()
    def start(self, context):
        # Collect all markdown urls and then crawl to get the data
        print("executed start")
        pass

    @hatchet.step(parents=["start"])
    def crawl(self, context):
        print("executed crawl")
        pass

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
