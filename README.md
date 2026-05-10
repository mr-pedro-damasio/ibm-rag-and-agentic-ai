# IBM RAG and Agentic AI

> Jupyter Notebooks for developing generative AI applications with RAG (Retrieval-Augmented Generation) and Agentic AI patterns, using OpenRouter for LLM access.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/anomalyco/ibm-rag-and-agentic-ai)

---

## Overview

This repository provides a hands-on learning environment for building generative AI applications. It includes Jupyter notebooks covering topics from basic LLM prompting to advanced RAG and agentic workflows.

---

## What's Included

- **Dev Container**: Python-based container with JupyterLab, Node.js LTS, Docker-in-Docker
- **AI Assistants**: Claude Code, GitHub Copilot, Gemini CLI, opencode — all pre-installed
- **Agent Instructions**: Structured guidance for AI tools via `CLAUDE.md` → `AGENTS.md` → `.github/copilot-instructions.md`
- **VS Code Config**: Recommended extensions — Python, Pylance, Jupyter, GitHub Copilot, Claude Code, Gemini Code Assist

---

## Getting Started

### Option 1 — GitHub Codespaces (recommended)

1. Click **Open in GitHub Codespaces** above, or go to **Code → Codespaces → New codespace**.
2. Wait for the environment to build (first run takes a few minutes).
3. Copy `.env.example` to `.env` and fill in your OpenRouter API key.
4. Open a notebook in `notebooks/` and start coding.

### Option 2 — Dev Container (local)

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop) and the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for VS Code.

1. Clone the repository.
2. Open the project in VS Code.
3. When prompted, click **Reopen in Container** (or run `Dev Containers: Reopen in Container` from the command palette).
4. Copy `.env.example` to `.env` and fill in your OpenRouter API key.
5. Open a notebook and start coding.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key for LLM access |

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
# Edit .env with your values
```

> **Never commit `.env` to version control.**

---

## Notebooks

The notebooks are organized under `notebooks/`:

- **Module 01** — Getting started with LLMs via OpenRouter, basic prompting, LangChain integration
- **Module 02** — (coming soon)

---

## Docker

Build and run the production Jupyter image:

```bash
# Build the production image
docker build -f docker/Dockerfile.prod -t jupyter-prod .

# Run with Docker Compose
docker compose -f docker/docker-compose.yml up
```

The production image uses `quay.io/jupyter/base-notebook` and serves JupyterLab on port 8888.

---

## AI Tools

The following AI coding assistants are pre-installed in the dev container:

| Tool | CLI | VS Code Extension |
|------|-----|-------------------|
| [Claude Code](https://claude.ai/code) | `claude` | Anthropic Claude Code |
| [GitHub Copilot](https://github.com/features/copilot) | `gh copilot` | GitHub Copilot |
| [Gemini CLI](https://github.com/google-gemini/gemini-cli) | `gemini` | Gemini Code Assist |
| [opencode](https://opencode.ai) | `opencode` | — |

---

## Development

### Python Dependencies

Required packages (installed automatically in the dev container):

- `python-dotenv` — environment variable management
- `langchain-openrouter` — LangChain integration with OpenRouter
- `rich` — rich terminal output for notebooks

To add new dependencies:

```bash
pip install <package>
```

### Adding new notebooks

1. Create a new directory under `notebooks/` following the module structure.
2. Add your `.ipynb` files.
3. Update this README with the new module description.

---

## License

[MIT](LICENSE)
