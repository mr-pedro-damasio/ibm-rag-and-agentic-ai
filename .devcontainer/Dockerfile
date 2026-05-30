ARG PROJECT_NAME=ibm-rag-and-agentic-ai
FROM mcr.microsoft.com/devcontainers/python:3
RUN pip install --upgrade pip && pip install \
    jupyterlab \
    notebook \
    python-dotenv \
    rich \
    pydantic \
    beautifulsoup4 \
    langchain \
    langchain-classic \
    langchain-core \
    langchain-community \
    langchain-text-splitters \
    langchain-openrouter \
    langchain-openai \
    langchain-chroma \
    chromadb \
    pypdf \
    langchain-experimental
ARG PROJECT_NAME
LABEL dev.containers.project="${PROJECT_NAME}-devcontainer"
