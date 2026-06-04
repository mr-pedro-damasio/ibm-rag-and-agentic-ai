# IBM RAG and Agentic AI

> Jupyter Notebooks for developing generative AI applications with RAG (Retrieval-Augmented Generation) and Agentic AI patterns, using OpenRouter for LLM access.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/anomalyco/ibm-rag-and-agentic-ai)

---

## Overview

This repository provides a hands-on learning environment for building generative AI applications. It includes Jupyter notebooks covering topics from basic LLM prompting to advanced RAG and agentic workflows.

---

## What's Included

- **Dev Container**: Python-based container with JupyterLab and Node.js LTS
- **AI Assistants**: Claude Code, GitHub Copilot, Antigravity CLI, opencode — all pre-installed
- **Agent Instructions**: Structured guidance for AI tools via `CLAUDE.md` → `AGENTS.md` → `.github/copilot-instructions.md`
- **VS Code Config**: Recommended extensions — Python, Pylance, Jupyter, GitHub Copilot, Claude Code

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

Organized under `notebooks/` by course and module:

### Course 01 — Develop Generative AI Applications: Get Started
- **Module 01** — Prompt engineering and LangChain PromptTemplates
- **Module 02** — Building smarter AI apps — empowering LLMs with LangChain (tools and agents)
- **Module 03** — Hands-on with GenAI: choosing the right model for your application

### Course 02 — Build RAG Applications: Get Started
- **Module 01** — Summarize private documents using RAG, LangChain, and LLMs
- **Module 02** — Gradio interfaces and QA bots with LangChain
- **Module 03** — AI icebreaker bot with LlamaIndex

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

All dependencies are declared in `requirements.txt` and installed automatically when the container is created. Key packages include:

- `jupyterlab`, `notebook` — interactive notebook environment
- `python-dotenv` — environment variable management
- `langchain`, `langchain-community`, `langchain-openrouter`, `langchain-openai`, `langchain-chroma` — LangChain ecosystem
- `pypdf` — PDF document loading
- `gradio` — interactive web UIs
- `huggingface_hub` — Hugging Face model and dataset access

To add a new dependency, add it to `requirements.txt` then run:

```bash
sudo uv pip install --system -r requirements.txt
```

### Adding new notebooks

1. Create a new directory under `notebooks/` following the course/module structure.
2. Add your `.ipynb` or `.py` files.
3. Update this README with the new module description.

---

## License

[MIT](LICENSE)
