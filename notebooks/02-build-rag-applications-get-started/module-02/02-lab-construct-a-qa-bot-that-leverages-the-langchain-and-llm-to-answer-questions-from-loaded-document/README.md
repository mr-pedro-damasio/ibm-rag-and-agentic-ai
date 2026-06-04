# Lab 02 — QA Bot with LangChain, RAG, and Gradio

A Retrieval-Augmented Generation (RAG) chatbot that accepts a PDF upload and answers questions about its content. Built with LangChain, ChromaDB, OpenRouter, and a Gradio web interface.

---

## Architecture

The UI is split into two explicit steps so the embedding cost is paid once per document, not once per question.

```
── Step 1: Process Document ──────────────────────────────────────
User uploads PDF
       │
       ▼
 PdfReader (pypdf)               ← reads raw pages
       │
       ▼
 RecursiveCharacterTextSplitter  ← splits into overlapping chunks
       │
       ▼
 OpenAIEmbeddings (via OpenRouter)  ← embeds each chunk
       │
       ▼
 Chroma (in-memory vector store)    ← stores embeddings
       │
       ▼
 vectordb.as_retriever()            ← cached in gr.State

── Step 2: Ask ───────────────────────────────────────────────────
User types question
       │
       ▼
 retriever (from gr.State)      ← similarity search, no re-embedding
       │
       ▼
 RAG_PROMPT (ChatPromptTemplate) ← explicit prompt with context + question
       │
       ▼
 ChatOpenRouter (GPT-4o-mini)   ← generates the final answer
       │
       ▼
 StrOutputParser → Gradio answer box
```

---

## Function Responsibilities

| Function | What it does |
|---|---|
| `llm_model()` | Instantiates `ChatOpenRouter` with the given parameters (called once at module load) |
| `document_loader(file_path)` | Reads a PDF with `pypdf.PdfReader`; returns a list of `Document` objects |
| `text_splitter(data, ...)` | Splits `Document` objects into chunks with configurable size and overlap |
| `build_vector_store(chunks)` | Embeds chunks and loads them into an in-memory Chroma store |
| `format_docs(docs)` | Joins retrieved `Document` objects into a single context string |
| `build_retriever(file)` | Orchestrates load → split → embed → store → retriever; stores result in `gr.State` |
| `answer_question(query, retriever_obj)` | Runs the LCEL chain against the cached retriever; returns the answer string |

---

## Dependencies

| Package | Role |
|---|---|
| `langchain` | Core orchestration primitives |
| `langchain-core` | LCEL primitives — `ChatPromptTemplate`, `RunnablePassthrough`, `StrOutputParser` |
| `langchain-openai` | `OpenAIEmbeddings` wrapper |
| `langchain-openrouter` | `ChatOpenRouter` LLM wrapper |
| `langchain-chroma` | Chroma vector store integration |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` |
| `gradio` | Web UI (`gr.Blocks` + `gr.State`) |
| `pypdf` | PDF parsing |
| `python-dotenv` | `.env` loading |

---

## Configuration

Non-secret configuration (model names, tuning parameters, Gradio settings, prompt template) lives in `config.py`. Secrets are the only things that belong in `.env`.

Copy `.env.example` to `.env` and fill in the two values:

**`.env` — secrets (two entries only)**

| Variable | Notes |
|---|---|
| `OPENROUTER_API_KEY` | Required — your OpenRouter API key |
| `OPENROUTER_BASE_URL` | Required — `https://openrouter.ai/api/v1` |

**`config.py` — everything else**

| Constant | Default | Notes |
|---|---|---|
| `LLM_MODEL` | `openai/gpt-4o-mini` | Chat model for answer generation |
| `LLM_TEMPERATURE` | `0.7` | Randomness; set to `0` for deterministic output |
| `LLM_MAX_TOKENS` | `2048` | Max tokens in the generated answer |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Embedding model |
| `EMBEDDING_DIMENSIONS` | `1024` | Embedding size (256–1536 supported) |
| `CHUNK_SIZE` | `1000` | Characters per text chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `GRADIO_SERVER_NAME` | `localhost` | Bind address |
| `GRADIO_SERVER_PORT` | `7860` | Web server port |
| `GRADIO_SHARE` | `False` | Set to `True` to create a public Gradio tunnel |

---

## How to Run

```bash
python qabot.py
```

Open `http://localhost:7860`:

1. Upload a PDF and click **Process Document** — wait for the status message confirming how many chunks were indexed.
2. Type a question and click **Ask** — the answer is generated from the cached retriever without re-embedding.

---

## Troubleshooting

**App fails to start: `EnvironmentError: Required environment variables not set`**
Copy `.env.example` to `.env` and fill in both values.

**Status box: `"No text could be extracted from this PDF"`**
The PDF is likely a scanned document (image pages, not digital text). pypdf can only read PDFs with an embedded text layer. Use a PDF exported from a word processor.

**Status box: `"Failed to process document: ... 404 ..."`**
OpenRouter's `/v1/embeddings` endpoint only covers models in their embeddings catalogue. Change `EMBEDDING_MODEL` in `config.py` from `"openai/text-embedding-3-small"` to `"text-embedding-3-small"` (drop the `openai/` prefix).

**Answer box: `"Failed to generate an answer: ... auth ..."`**
Your `OPENROUTER_API_KEY` is set but invalid. Check your key at openrouter.ai/keys.
