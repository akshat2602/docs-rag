# Building a Robust scalable RAG System with LangChain, FastAPI, Hatchet

## Introduction
Retrieval-Augmented Generation (RAG) is currently a hot trend, with hundreds of new startups relying heavily on RAG systems. New vector databases, embedding models, and multi-modal models are being developed and can be maximized effectively through RAG. While there are numerous tutorials and walkthroughs available that demonstrate building basic RAG applications, these are often not scalable or durable in the long run. Many of these tutorials are built on Jupyter Notebooks, which are great for prototyping but not ideal for production-level applications.

To bridge this gap, I'll guide you through building a scalable and durable RAG application designed to facilitate chat interactions with your documentation. The tools we'll be using include:
- **LangChain**: For interacting with embedding models and the vector store.
- **FastAPI**: Serving as the gateway to interact with our RAG application.
- **Hatchet**: A distributed task queue/workflow orchestrator for queuing embedding and crawling tasks, ensuring durability in our stack.
- **ChromaDB**: Our choice of vector store.

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
