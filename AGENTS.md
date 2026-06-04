# Agents.md

## Project
IBM RAG and Agentic AI — Jupyter Notebooks for learning generative AI development with RAG and agentic patterns.

## Dev Environment
- Base image: `mcr.microsoft.com/devcontainers/python:3`
- Python packages: see `requirements.txt`
- Node.js LTS (for CLI tools: Claude Code, opencode, Antigravity CLI, GitHub Copilot CLI)
- Setup script: `.devcontainer/setup.sh` (runs on container creation)

## Package Management
- Dependencies are declared in `requirements.txt` and installed via `uv pip install --system`
- To add a new package: add it to `requirements.txt`, then run `sudo uv pip install --system -r requirements.txt`
- Rebuild the container to apply changes persistently

## Required Environment Variables
- `OPENROUTER_API_KEY` — Set in `.env` (copied from `.env.example`)

## Notebooks
Organized under `notebooks/` by course and module:

### Course 01 — Develop Generative AI Applications: Get Started
- **Module 01**: Prompt engineering and LangChain PromptTemplates
- **Module 02**: Building smarter AI apps — empowering LLMs with LangChain (tools and agents)
- **Module 03**: Hands-on with GenAI — choosing the right model for your application

### Course 02 — Build RAG Applications: Get Started
- **Module 01**: Summarize private documents using RAG, LangChain, and LLMs
- **Module 02**: Gradio interfaces and QA bots with LangChain
- **Module 03**: AI icebreaker bot with LlamaIndex
