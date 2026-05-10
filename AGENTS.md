# Agents.md

## Project
Jupyter Notebooks devcontainer — Python with Node.js (for CLI tools).

## Dev Environment
- Base: `mcr.microsoft.com/devcontainers/python:3`
- JupyterLab + notebook installed via pip
- VS Code extensions: Python, Pylance, Jupyter

## Package Management
- pip (no requirements.txt yet — add as needed)
- `pip install <package>`

## Production
- Build: `docker build -f docker/Dockerfile.prod -t jupyter-prod .`
- Run: `docker compose -f docker/docker-compose.yml up`
- Image: `quay.io/jupyter/base-notebook`, no auth, port 8888
