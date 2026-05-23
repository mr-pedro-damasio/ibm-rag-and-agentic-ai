ARG PROJECT_NAME=ibm-rag-and-agentic-ai
FROM mcr.microsoft.com/devcontainers/python:3
RUN pip install --upgrade pip && pip install jupyterlab notebook python-dotenv langchain-openrouter rich langchain_community pypdf langchain_openai chromadb langchain
ARG PROJECT_NAME
LABEL dev.containers.project="${PROJECT_NAME}"
