ARG PROJECT_NAME=zero-config-devcontainer
FROM mcr.microsoft.com/devcontainers/python:3
RUN pip install --upgrade pip && pip install jupyterlab notebook
ARG PROJECT_NAME
LABEL dev.containers.project="${PROJECT_NAME}"
