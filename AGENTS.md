# Agents.md

## Project
IBM RAG and Agentic AI — Jupyter Notebooks for learning generative AI development with RAG and agentic patterns.

## Dev Environment
- Base: `mcr.microsoft.com/devcontainers/python:3`
- Python packages: jupyterlab, notebook, python-dotenv, langchain-openrouter, rich
- Node.js LTS (for CLI tools: Claude Code, opencode, Gemini CLI)

## Package Management
- pip (no requirements.txt — packages installed via Dockerfile)
- `pip install <package>` to add new dependencies
- Update `.devcontainer/Dockerfile` if packages should be pre-installed

## Required Environment Variables
- `OPENROUTER_API_KEY` — Set in `.env` (copied from `.env.example`)

## Notebooks
- Located under `notebooks/`
- Module 01: LLM prompting with OpenRouter + LangChain
- Module 02: (coming soon)

## Production
- Build: `docker build -f docker/Dockerfile.prod -t jupyter-prod .`
- Run: `docker compose -f docker/docker-compose.yml up`
- Image: `quay.io/jupyter/base-notebook`, no auth, port 8888
