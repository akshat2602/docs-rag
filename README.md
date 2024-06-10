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

Just as a sanity check, our directory structure must look something like this - 
![worker-directory-structure](assets/worker-directory-structure.png)

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
If we've set our environment variables correctly, our worker should startup. We can check this by navigating to our Hatchet admin dashboard and to the workers page where we'll see a new worker pop up. You can also access the same page by accessing this route using your browser: http://localhost:8080/workers.

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

To set up the API, we'll be using FastAPI and running the API server using Poetry. Let's begin by creating a new directory called backend in the root directory of our project:
```bash
mkdir backend
```
Now, let's initialize a Poetry project inside the backend folder:
```bash
cd backend
poetry init
```
Similar to the workers directory, we won't set up dependencies right now. We'll do that using the following command: 
```bash
poetry add fastapi \
uvicorn \
openai \
tiktoken \
langchain \
langchain-chroma \
pydantic-settings \
langchain-openai \
langchain-community \
chromadb
```
If the command runs successfully, you'll see a new `poetry.lock` file inside the `backend` directory and new dependencies in your `pyproject.toml` file.

We'll create a `.env` file to store our tokens inside the backend directory:

```dotenv
DOMAIN=localhost

ENVIRONMENT=local

PROJECT_NAME=docs-rag

OPENAI_API_KEY=

CHROMA_COLLECTION=langchain
CHROMA_HOST=chromadb
CHROMA_PORT=8000
```
You can set the same OpenAI API token in the `backend` directory as the one in the `workers` one.

Once that is done, we can start defining our API by loading in the `.env` file and starting up a listener.
To get started, create an `app` directory inside the `backend` directory. Inside this `app` directory, create a `core` directory that'll house our database connection, our backend config and OpenAI connection.

Inside the `core` directory, create three new files called `config.py`, `db.py` and `openai.py`. Make sure to also add in a `__init__.py` in the `app` and `core` directory to help python recognize the as modules. 

Our directory structure will look something like this - 
![backend-directory-structure](assets/backend-directory-structure.png)

`config.py` will have the following contents - 
```python
from typing import Literal
from pydantic import AnyUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )
    API_V1_STR: str = "/api/v1"
    DOMAIN: str = "localhost"
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"

    OPENAI_API_KEY: str = "your-openai-api-key"

    BACKEND_CORS_ORIGINS: list[AnyUrl] | AnyUrl = [
        "http://localhost",
        "http://localhost:5000",
        "http://localhost:5173",
        "https://localhost",
        "https://localhost:5173",
        "https://localhost:5000",
        "http://localhost.tiangolo.com",
        "http://localhost",
        "http://localhost:5173",
        "https://localhost",
        "https://localhost:5173",
        "http://localhost.tiangolo.com",
    ]

    PROJECT_NAME: str
    CHROMA_COLLECTION: str = "langchain"
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000


settings = Settings()  # type: ignore

```
This code defines a Pydantic Settings model for managing application settings of our FastAPI. It load settings from an environment file (".env") while ignoring empty values.

The `db.py` file will house our DB connection to chroma and store a DB object that can be used to perform DB operations. The file will have the following DB connection code - 
```python
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

```
- The `init_db()` function initializes the database by creating an instance of Chroma with settings from the application configuration (settings). It sets up embeddings using OpenAIEmbeddings and a ChromaDB client.
- The `get_db()` function returns the initialized database instance.
- The `get_relevant_documents()` function takes a question as input, retrieves relevant documents from the database using a retriever configured for Maximal Marginal Relevance (MMR) search, and returns a list of relevant documents.

The `openai.py` file will house the code for establishing a connection to ChatGPT and have the prompt for getting answers - 
```python
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

```
- The `init_openai()` function initializes the OpenAI model using the API key from the application settings.
- The `get_llm()` function returns the initialized OpenAI model instance.
- The `answer_question()` function fetches relevant documents from the database, prepares a prompt with the question and relevant documents, invokes the OpenAI model with the prompt, and returns the generated answer along with the relevant documents.
- The `get_magic_prompt()` function returns a formatted prompt template that includes placeholders for relevant documents and the question. This prompt template is used to structure the input to the OpenAI model.

Now that we've defined the core components of our backend API, we can now define our routes for our server. Create a `api` directory inside `app` directory. In this `api` directory, create a file called `main.py` which will house our route to answer a question 

```python
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
    return {"answer": answer, "sources": sources}

```
The answer route takes in a `QuestionRequest` model and calls the `answer_question` function we've defined in the `openai.py` file which'll give us an answer and a list of sources which we can return by serializing it using the `QuestionResponse` model.

We can now define our FastAPI listener in a `main.py` file in our `app` directory.
```python
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.main import api_router
from app.core.config import settings
from app.core.db import init_db
from app.core.openai import init_openai


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code here
    init_db()
    init_openai()
    yield
    # Cleanup code here


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)
```
Here we set up a FastAPI application with middleware for CORS (Cross-Origin Resource Sharing) handling and includes the API routes defined in app.api.main.api_router.

- An asynchronous context manager lifespan() is defined to handle startup and cleanup tasks for the application. Inside the lifespan() context manager, init_db() and init_openai() functions are called to initialize the database connection and the OpenAI model.
- The FastAPI application is instantiated with the project name from the settings and an OpenAPI endpoint is configured.
- CORS middleware is added to allow cross-origin requests from any origin with any method and headers.
- The API routes defined in api_router are included in the application with the specified prefix.

The directory structure will now look something like this - 
![final-directory-structure](assets/final-directory-structure.png)

We can start our API server by running this command inside the `backend` directory - 
```bash
poetry run fastapi dev app/main.py --host 0.0.0.0 --port 5000
```
If everything is setup correctly, FastAPI will startup an API server. To check, you can navigate to http://localhost:5000/docs.
It'll look something like this - 
![API-Docs](assets/API-Docs.png)

You can test out the API by clicking on `Try it out` and asking a question. To trigger an API call just click on `Exectue` and an API call will be sent. An example API call - 
![API-response](assets/API-response.png)

## Conclusion
In conclusion, we've built a scalable and durable RAG application using LangChain, FastAPI, Hatchet and ChromaDB. You can check out the code on the [GitHub repository](https://github.com/akshat2602/docs-rag)

You can checkout documentation for all of the above tools here - 
1) Hatchet: https://docs.hatchet.run
2) FastAPI: https://fastapi.tiangolo.com/learn/
3) LangChain: https://python.langchain.com/v0.2/docs/introduction/
