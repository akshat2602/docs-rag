# Building a Robust scalable RAG System with LangChain, FastAPI, Hatchet

## Introduction
Retrieval-Augmented Generation (RAG) is currently a hot trend, with hundreds of new startups relying heavily on RAG systems. New vector databases, embedding models, and multi-modal models are being developed and can be maximized effectively through RAG. While there are numerous tutorials and walkthroughs available that demonstrate building basic RAG applications, these are often not scalable or durable in the long run. Many of these tutorials are built on Jupyter Notebooks, which are great for prototyping but not ideal for production-level applications.

To bridge this gap, I'll guide you through building a scalable and durable RAG application designed to facilitate chat interactions with your documentation. For this walkthrough we'll be using Hatchet's documentation as an example. The tools we'll be using include:
- **LangChain**: For interacting with embedding models and the vector store.
- **FastAPI**: Serving as the gateway to interact with our RAG application.
- **Hatchet**: A distributed task queue/workflow orchestrator for queuing embedding and crawling tasks, ensuring durability in our stack.
- **ChromaDB**: Our choice of vector store.

This walkthrough assumes some familiarity with Python and how Retrieval-Augmented Generation (RAG) works.

## Overview:
### LangChain
LangChain is a framework for developing applications powered by large language models (LLMs). It provides useful abstractions over widely used services needed for building context-aware reasoning applications or RAG systems.

### FastAPI
FastAPI is a modern, fast (high-performance) web framework for building APIs with Python, based on standard Python type hints. It is known for its ease of use and automatic interactive API documentation.

### Hatchet
Hatchet is a distributed, fault-tolerant task queue designed to replace traditional message brokers and pub/sub systems. It addresses problems related to concurrency, fairness, and durability, making it ideal for handling complex workflows.

## Setting Up the Development Environment

### Prerequisites:
1. **Docker and Docker Compose**:
    - Install Docker and Docker Compose following the official Docker documentation: [Docker Installation Guide](https://docs.docker.com/engine/install/)
2. **Poetry (optional)**:
    - Poetry is a tool for dependency management and packaging in Python. You can find the installation instructions here: [Poetry Documentation](https://python-poetry.org/docs/)
3. **A code editor of your choice**
4. **GitHub account**:
    - For setting up a token to access the API.
5. **Developer OpenAI account/LLM endpoint**

## Building the RAG System
Before a we start setting up the project, we'll initialize a git repository to track our changes and so that we don't lose our code. 
```bash
git init
```
After this create a new file called `.gitignore` with the following contents to make sure you don't accidentally commit any sensitive information or cache files or our database:
```
.env
.idea
.fleet

themes

# Django stuff
*.log
*.pot
*.sqlite3
static/


# Unit test / coverage reports
.tox/
.coverage
.cache
nosetests.xml
coverage.xml

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]

# Mac stuff
.DS_Store

chroma-data
```

### Setup a new project directory
```bash
mkdir docs-rag
cd docs-rag
```

We'll be self-hosting Hatchet for this walkthrough, but you're free to use Hatchet Cloud if you prefer.

### Self hosting hatchet
We'll be modifying the hatchet self-hosting Docker Compose file provided by Hatchet, which you can find [here](https://docs.hatchet.run/self-hosting/docker-compose)

We'll be adding ChromaDB to this compose file as our vector store. Create a new empty Docker Compose file in your project directory and a new directory that will house our Hatchet config. Inside our Hatchet directory, we'll store the Caddyfile mentioned in the self-hosting docs. The modified Docker Compose file will look like this:
```yaml
version: "3.8"
services:
  postgres:
    image: postgres:15.6
    command: postgres -c 'max_connections=200'
    restart: always
    hostname: "postgres"
    environment:
      - POSTGRES_USER=hatchet
      - POSTGRES_PASSWORD=hatchet
      - POSTGRES_DB=hatchet
    ports:
      - "5435:5432"
    volumes:
      - hatchet_postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: [ "CMD-SHELL", "pg_isready", "-d", "hatchet" ]
      interval: 10s
      timeout: 10s
      retries: 5
      start_period: 10s
  rabbitmq:
    image: "rabbitmq:3-management"
    hostname: "rabbitmq"
    ports:
      - "5673:5672" # RabbitMQ
      - "15673:15672" # Management UI
    environment:
      RABBITMQ_DEFAULT_USER: "user"
      RABBITMQ_DEFAULT_PASS: "password"
    volumes:
      - "hatchet_rabbitmq_data:/var/lib/rabbitmq"
      - "hatchet_rabbitmq.conf:/etc/rabbitmq/rabbitmq.conf" # Configuration file mount
    healthcheck:
      test: [ "CMD", "rabbitmqctl", "status" ]
      interval: 10s
      timeout: 10s
      retries: 5
  migration:
    image: ghcr.io/hatchet-dev/hatchet/hatchet-migrate:latest
    environment:
      DATABASE_URL: "postgres://hatchet:hatchet@postgres:5432/hatchet"
    depends_on:
      postgres:
        condition: service_healthy
  setup-config:
    image: ghcr.io/hatchet-dev/hatchet/hatchet-admin:latest
    command: /hatchet/hatchet-admin quickstart --skip certs --generated-config-dir /hatchet/config --overwrite=false
    environment:
      DATABASE_URL: "postgres://hatchet:hatchet@postgres:5432/hatchet"
      DATABASE_POSTGRES_PORT: "5432"
      DATABASE_POSTGRES_HOST: "postgres"
      SERVER_TASKQUEUE_RABBITMQ_URL: amqp://user:password@rabbitmq:5672/
      SERVER_AUTH_COOKIE_DOMAIN: localhost:8080
      SERVER_AUTH_COOKIE_INSECURE: "t"
      SERVER_GRPC_BIND_ADDRESS: "0.0.0.0"
      SERVER_GRPC_INSECURE: "t"
      SERVER_GRPC_BROADCAST_ADDRESS: localhost:7077
    volumes:
      - hatchet_certs:/hatchet/certs
      - hatchet_config:/hatchet/config
    depends_on:
      migration:
        condition: service_completed_successfully
      rabbitmq:
        condition: service_healthy
      postgres:
        condition: service_healthy
  hatchet-engine:
    image: ghcr.io/hatchet-dev/hatchet/hatchet-engine:latest
    command: /hatchet/hatchet-engine --config /hatchet/config
    restart: on-failure
    depends_on:
      setup-config:
        condition: service_completed_successfully
      migration:
        condition: service_completed_successfully
    ports:
      - "7077:7070"
    environment:
      DATABASE_URL: "postgres://hatchet:hatchet@postgres:5432/hatchet"
      SERVER_GRPC_BIND_ADDRESS: "0.0.0.0"
      SERVER_GRPC_INSECURE: "t"
    volumes:
      - hatchet_certs:/hatchet/certs
      - hatchet_config:/hatchet/config
  hatchet-api:
    image: ghcr.io/hatchet-dev/hatchet/hatchet-api:latest
    command: /hatchet/hatchet-api --config /hatchet/config
    restart: on-failure
    depends_on:
      setup-config:
        condition: service_completed_successfully
      migration:
        condition: service_completed_successfully
    environment:
      DATABASE_URL: "postgres://hatchet:hatchet@postgres:5432/hatchet"
    volumes:
      - hatchet_certs:/hatchet/certs
      - hatchet_config:/hatchet/config
  hatchet-frontend:
    image: ghcr.io/hatchet-dev/hatchet/hatchet-frontend:latest
  caddy:
    image: caddy:2.7.6-alpine
    ports:
      - 8080:8080
    volumes:
      - ./hatchet/Caddyfile:/etc/caddy/Caddyfile
  chromadb:
    image: chromadb/chroma
    environment:
      - IS_PERSISTENT=TRUE
    volumes:
      # Default configuration for persist_directory in chromadb/config.py
      # Currently it's located in "/chroma/chroma/"
      - ./chroma-data:/chroma/chroma/
    ports:
      - 8000:8000

volumes:
  hatchet_postgres_data:
  hatchet_rabbitmq_data:
  hatchet_rabbitmq.conf:
  hatchet_config:
  hatchet_certs:
``` 
Inside our hatchet directory, create a new Caddyfile with the name `Caddyfile` and configure it as follows:
```
http://localhost:8080 {
	handle /api/* {
		reverse_proxy hatchet-api:8080
	}

	handle /* {
		reverse_proxy hatchet-frontend:80
	}
}
```
Your current directory structure will look something like this - 
![hatchet-directory](assets/hatchet-directory.png)
To test out our hatchet config, run the command - 
```bash
docker compose -f docker-compose.hatchet.yml up -d
```
Try to access http://localhost:8080 through your browser. If you see the login screen, you have successfully self-hosted Hatchet.
We'll now move on to setting up our embedding and crawling workflows and workers for Hatchet.

### Integrating Hatchet
Create a new directory called `workers` in your project directory and initialize a new Poetry project using the command - 
```bash
cd workers
poetry init
```
Walk through the prompts to finish initializing a Poetry project. You can answer "no" to all the dependency-related questions as we'll be setting up our dependencies in a few moments.

You can install the required dependencies for LangChain, OpenAI, and Hatchet using the following command inside the `workers` directory:
```bash
poetry add hatchet-sdk \
langchain \
langchain-chroma \
langchain-community \
openai \
langchain-openai
```
If the command runs successfully, your pyproject.toml will look like the following, and a new poetry.lock file will be created inside the workers directory:
```toml
[tool.poetry.dependencies]
python = ">=3.10,<3.13"
hatchet-sdk = "^0.26.3"
langchain = "^0.2.3"
langchain-chroma = "^0.1.1"
langchain-community = "^0.2.4"
openai = "^1.33.0"
langchain-openai = "^0.1.8"
```

### Setting up workers:
Create a .env file inside the workers directory that will house our GitHub, Hatchet, and OpenAI tokens along with ChromaDB credentials for accessing the services. The .env file will have the following contents:

```dotenv
HATCHET_CLIENT_TOKEN=""
HATCHET_CLIENT_TLS_STRATEGY=none

GITHUB_PERSONAL_ACCESS_TOKEN=

OPENAI_API_KEY=

CHROMA_COLLECTION=langchain
CHROMA_HOST=localhost
CHROMA_PORT=8000
```
You can create a .env.example file to be committed to your repository with empty tokens as an example config file with the above contents.
#### Generating required tokens: 

- **Hatchet Token**: To create a token for Hatchet, follow the steps listed [here](https://docs.hatchet.run/self-hosting/docker-compose#access-the-hatchet-ui)
- **GitHub Token**: To create a GitHub access token, follow the steps listed [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token)
- **OpenAI API Token**: You can create an OpenAI API token [here](https://platform.openai.com/api-keys)


With these tokens in place, you will be able to configure and run your workers effectively.

Now that the environment variables have been set up for our workers, we can start defining workflows. Our workflow will operate as follows:

We'll trigger a workflow that will crawl the documentation and push for an embedding workflow when an event called `rag:crawl` is triggered.

The `crawl` step in our `rag:crawl` workflow will fetch all possible markdown files (typically where documentation is defined for large open-source projects). The `push_for_embed` step in our workflow will iterate over all the documentation files and push each one iteratively for an embedding workflow by triggering an event called `rag:embeddings`.

The `rag:embeddings` workflow has two steps as well. The first step, `fetch_document`, fetches the contents of the file it received as an input for the event. After fetching the contents, the next step, `store_embeddings`, creates embeddings of the contents and stores them in our ChromaDB vector store.

This approach makes our RAG system scalable and durable as we're distributing the crawling, embedding, and indexing workload across different workers and organizing the tasks into steps, making the entire process replayable.

To start, create a `main.py` file in the `workers` directory where we'll initialize connections to OpenAI API and our chromaDB instance that we had started as part of our Hatchet Docker Compose file. 
```python
from hatchet_sdk import Hatchet
from dotenv import load_dotenv
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
```
Once that is done, we'll define our workflows, first the `rag:crawl` workflow: 
```python
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
        return {"status": "pushed for embeddings"}
```
Here's what each step in the workflow does - 
1. **crawl**: This step uses a `GithubFileLoader` to collect all markdown URLs from the specified repository (`hatchet-dev/hatchet`). It filters for files ending with `.md` or `.mdx` within the `frontend/docs` directory and retrieves their paths. The collected file paths are returned with a status message.

2. **push_for_embed**: This step, which depends on the `crawl` step, takes the file paths obtained from the previous step. It logs each file path and pushes an event (`rag:embeddings`) to Hatchet's client, indicating that these files are ready for embedding processing. This step also returns a status message indicating that embeddings have been pushed.

Now that our `rag:crawl` workflow is defined, we'll define the `rag:embeddings` workflow which'll create embeddings for each indiviual document.
```python
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
```
1. **fetch_document**: This step, with a timeout of 5 minutes, downloads a document from a GitHub repository using the provided file path from the workflow input. It logs the download process and returns the document content along with a status message.

2. **store_embeddings**: This step, which depends on the `fetch_document` step and has a timeout of 10 minutes, processes the downloaded document to create embeddings. It logs the embedding process, splits the document into smaller chunks using `RecursiveCharacterTextSplitter`, and stores these chunks in the chromaDB instance we had previously initialized. Finally, it logs the completion of the embedding process and returns a status message.

To finish the worker creation, we'll create a Hatchet worker, register these workflows and start our worker - 
```python
worker = hatchet.worker("docs-rag-worker")
worker.register_workflow(RAGCrawlerWorkflow())
worker.register_workflow(RAGEmbeddingsWorkflow())


worker.start()
```

We can run our worker in a new terminal inside the `workers` directory with the command - 
```bash
poetry run python -m main.py
```
If we've set our environment variables correctly, our worker should startup. We can check this by navigating to our Hatchet admin dashboard and to the workers page where we'll see a new worker pop up. You can also access the same page by accessing this route using your browser: http://localhost:8080/workers
The page will look something like this - 
![hatchet-worker](assets/hatchet-worker.png)
You can also navigate to the workflows page where you'll see the two workflows we've defined show up -
![hatchet-workflow](assets/hatchet-workflow.png)

You can test the workflows by manually triggering the RAGCrawlWorkflow using the dashboard. To do this, navigate to the RAGCrawlWorkflow page by clicking on the `View Workflow` button on the workflow page.
We can trigger the workflow by clicking the `Trigger Workflow` button where we can keep the input blank. 
The `RAGCrawlWorkflow` will finish relatively quickly and you can check the status and individual steps' data once it finishes. It'll look something like this - 
![crawl-workflow](assets/crawl-workflow.png)

The `RAGCrawlWorkflow` will trigger subsequent `RAGEmbeddingWorkflow` runs which you can see running by navigating to the `Workflow Runs` page from the side bar. 
![embedding-workflow](assets/embedding-workflow.png)
You can check out the logs and details about each individual run by clicking on the runs on the page.

We've successfully crawled, embedded and indexed our documentation, now we can move on to building the API that'll serve answers to user questions about Hatchet's documentation.

### Setting up the API:

