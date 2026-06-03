ARG PROJECT_NAME=ibm-rag-and-agentic-ai
FROM mcr.microsoft.com/devcontainers/python:3
RUN pip install --upgrade pip && pip install \
    jupyterlab \
    notebook \
    python-dotenv \
    wget \
    textloader \
    langchain \
    langchain-text-splitters \ 
    langchain-community \
    langchain-openrouter \
    langchain-openai \
    langchain-chroma \
    langchain-classic \
    # rich \
    # pydantic \
    # beautifulsoup4 \
    # langchain \
    # chromadb \
    pypdf \ 
    # langchain-experimental \
    gradio \
    huggingface_hub
ARG PROJECT_NAME
LABEL dev.containers.project="${PROJECT_NAME}-devcontainer"
