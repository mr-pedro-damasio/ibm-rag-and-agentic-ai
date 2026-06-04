# QA Bot — Improvement Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden and organise the QA bot by separating config from application logic, adding error handling, caching the LLM, consolidating redundant functions, and making comments explain *why* rather than *what*.

**Architecture:** All non-secret configuration moves to `config.py` (following the module-03 pattern). `.env` keeps exactly two entries: `OPENROUTER_API_KEY` and `OPENROUTER_BASE_URL`. Four independent batches — each verifiable before starting the next.

**Tech Stack:** Python, LangChain Core (LCEL), ChromaDB, pypdf, OpenAIEmbeddings, ChatOpenRouter, Gradio

---

## Issues Being Addressed

| # | Batch | Issue |
|---|-------|-------|
| 1 | 1 | All config is scattered inside `01-qabot.py` — no single place to tune model, chunk, or server settings |
| 2 | 1 | Magic number `1024` (embedding dimensions) hardcoded at call site, disconnected from the model name |
| 3 | 1 | Temperature `0.7` and `max_tokens=2048` are unnamed literals inside `answer_question` |
| 4 | 1 | `GRADIO_SERVER_NAME`, `GRADIO_SERVER_PORT`, `GRADIO_SHARE` mixed into `01-qabot.py` alongside business logic |
| 5 | 1 | `GRADIO_SHARE` reads from env — inconsistent with the rule that `.env` holds only secrets |
| 6 | 1 | `.env.example` has a `GRADIO_SHARE` entry that should not be there |
| 7 | 2 | No startup validation — if `OPENROUTER_API_KEY` / `OPENROUTER_BASE_URL` are missing the first API call fails with a cryptic auth error |
| 8 | 2 | No `try/except` in `build_retriever` — pypdf errors and API failures show raw Python tracebacks in the Gradio UI |
| 9 | 2 | No `try/except` in `answer_question` — LLM errors show raw Python tracebacks in the Gradio UI |
| 10 | 2 | Image-only PDFs extract all pages to `""`, produce 0 chunks; `Chroma.from_documents([])` fails with no user message |
| 11 | 3 | `llm_model()` re-instantiated on every question — new `ChatOpenRouter` object per query for no reason |
| 12 | 3 | `embedding_model()` is a single-caller pass-through wrapper — its parameter list is duplicated in `vector_database()` |
| 13 | 4 | Section markers use `##` (double hash) — unconventional Python; all describe *what* instead of *why* |
| 14 | 4 | The LCEL chain has no explanatory comment — non-obvious to someone learning RAG |
| 15 | 4 | README has no troubleshooting section for the most common failure points |

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `config.py` | **Create** | Single source of truth for all non-secret config: API settings, model names, tuning params, Gradio settings, RAG prompt template |
| `01-qabot.py` | **Modify** | Import from `config`; remove inline constants; add error handling; cache LLM; consolidate functions; rewrite comments |
| `.env.example` | **Modify** | Remove `GRADIO_SHARE` line — only `OPENROUTER_API_KEY` and `OPENROUTER_BASE_URL` belong here |
| `README.md` | **Modify** | Add Troubleshooting section; update Configuration table to mention `config.py` |

---

## Batch 1 — Create `config.py`, clean up `.env.example` and `01-qabot.py`

**Closes:** issues #1–6  
**Breaking changes:** None — same runtime behaviour

### What and why

Following the module-03 pattern: `config.py` loads `.env`, reads the two secrets via `os.getenv()`, then defines every other setting as a plain constant. `01-qabot.py` imports from `config` and contains no configuration of its own. `.env` and `.env.example` are trimmed to exactly two entries.

---

### Task 1: Create `config.py`

**Files:**
- Create: `config.py`

- [ ] **Step 1: Create the file**

  ```python
  import os
  import dotenv
  dotenv.load_dotenv()

  # Secrets — values come from .env, never hardcoded
  OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
  OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")

  # LLM used for answer generation
  LLM_MODEL = "openai/gpt-4o-mini"
  LLM_TEMPERATURE = 0.7   # slight randomness for natural-sounding answers; set to 0 for deterministic output
  LLM_MAX_TOKENS = 2048   # generous cap — RAG answers are rarely longer than a few paragraphs

  # Embedding model — routes through OpenRouter's /v1/embeddings endpoint
  # If this returns a 404 at runtime, try "text-embedding-3-small" (drop the "openai/" prefix)
  EMBEDDING_MODEL = "openai/text-embedding-3-small"
  EMBEDDING_DIMENSIONS = 1024  # text-embedding-3-small supports 256–1536; 1024 balances quality and cost

  # Text splitting — controls how the PDF is chunked before embedding
  CHUNK_SIZE = 1000   # characters per chunk; sized to fit comfortably within the model's context window
  CHUNK_OVERLAP = 50  # small overlap so a sentence split at a chunk boundary doesn't lose context

  # Gradio server
  GRADIO_SERVER_NAME = "localhost"
  GRADIO_SERVER_PORT = 7860
  GRADIO_SHARE = False  # set to True to create a public Gradio tunnel (use with caution)

  # RAG prompt — instructs the LLM to answer only from the retrieved context
  RAG_PROMPT_TEMPLATE = """
  You are a helpful assistant. Answer the question using only the context below.
  If the answer is not in the context, say "I don't know based on the provided document."
  Context:
  {context}
  Question: {question}
  """
  ```

- [ ] **Step 2: Verify the file is importable**

  ```bash
  cd notebooks/02-build-rag-applications-get-started/module-02/02-lab-construct-a-qa-bot-that-leverages-the-langchain-and-llm-to-answer-questions-from-loaded-document && \
  python -c "import config; print('config OK:', config.LLM_MODEL, config.EMBEDDING_MODEL)"
  ```
  Expected: `config OK: openai/gpt-4o-mini openai/text-embedding-3-small`

---

### Task 2: Update `01-qabot.py` — replace inline constants with imports from `config`

**Files:**
- Modify: `01-qabot.py`

- [ ] **Step 1: Replace the imports and constants block**

  Remove everything from `import os` through the last constant (`GRADIO_SHARE`) and replace with:

  ```python
  import logging
  from config import (
      OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
      LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
      EMBEDDING_MODEL, EMBEDDING_DIMENSIONS,
      CHUNK_SIZE, CHUNK_OVERLAP,
      GRADIO_SERVER_NAME, GRADIO_SERVER_PORT, GRADIO_SHARE,
      RAG_PROMPT_TEMPLATE,
  )
  from langchain_openrouter import ChatOpenRouter
  from pypdf import PdfReader
  from langchain_core.documents import Document
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_core.runnables import RunnablePassthrough
  from langchain_core.output_parsers import StrOutputParser
  from langchain_text_splitters import RecursiveCharacterTextSplitter
  from langchain_openai import OpenAIEmbeddings
  from langchain_chroma import Chroma
  import gradio as gr

  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
      datefmt="%H:%M:%S",
  )
  logger = logging.getLogger(__name__)

  RAG_PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
  ```

  Note: `load_dotenv()` moves to `config.py`, so the explicit call is removed from `01-qabot.py`.

- [ ] **Step 2: Update `text_splitter` to use config constants as defaults**

  ```python
  def text_splitter(data_to_split, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
  ```

- [ ] **Step 3: Update `build_retriever` to use config constants**

  Replace the `vector_database(...)` call:
  ```python
  # Before:
  vectordb = vector_database(chunks, OPENROUTER_EMBEDDINGS_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 1024)

  # After:
  vectordb = vector_database(chunks, EMBEDDING_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, EMBEDDING_DIMENSIONS)
  ```

- [ ] **Step 4: Update `answer_question` to use config constants**

  Replace the `llm_model(...)` call:
  ```python
  # Before:
  llm = llm_model(
      model_name=OPENROUTER_MODEL,
      ...
      model_temperature=0.7,
      model_maxtokens=2048,
  )

  # After:
  llm = llm_model(
      model_name=LLM_MODEL,
      model_api_key=OPENROUTER_API_KEY,
      model_base_url=OPENROUTER_BASE_URL,
      model_temperature=LLM_TEMPERATURE,
      model_maxtokens=LLM_MAX_TOKENS,
  )
  ```

- [ ] **Step 5: Verify — no raw config literals remain in `01-qabot.py`**

  ```bash
  grep -n "os\.getenv\|load_dotenv\|gpt-4o\|text-embedding\|0\.7\|2048\b\|1024\b\|chunk_size=1000\|chunk_overlap=50\|\"localhost\"\|7860" \
    notebooks/02-build-rag-applications-get-started/module-02/02-lab-construct-a-qa-bot-that-leverages-the-langchain-and-llm-to-answer-questions-from-loaded-document/01-qabot.py
  ```
  Expected: no output

---

### Task 3: Clean up `.env.example`

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Remove the `GRADIO_SHARE` line**

  `.env.example` should contain exactly:
  ```
  # Copy this file to .env and fill in the values.
  # Never commit .env to version control.

  OPENROUTER_API_KEY=your_openrouter_api_key_here
  OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
  ```

- [ ] **Step 2: Verify**

  ```bash
  grep -v "^#\|^$" .env.example
  ```
  Expected: exactly two lines — `OPENROUTER_API_KEY=...` and `OPENROUTER_BASE_URL=...`

- [ ] **Step 3: Commit Batch 1**

  ```bash
  git add \
    notebooks/02-build-rag-applications-get-started/module-02/02-lab-construct-a-qa-bot-that-leverages-the-langchain-and-llm-to-answer-questions-from-loaded-document/config.py \
    notebooks/02-build-rag-applications-get-started/module-02/02-lab-construct-a-qa-bot-that-leverages-the-langchain-and-llm-to-answer-questions-from-loaded-document/01-qabot.py \
    .env.example
  git commit -m "refactor(qabot): extract all config to config.py, trim .env to secrets only"
  ```

---

## Batch 2 — Startup Validation and Error Handling

**Closes:** issues #7–10  
**Breaking changes:** None — happy-path behaviour is identical; only error paths change

### What and why

Gradio does not pretty-print Python tracebacks — they appear as a jarring red block. Wrapping the two Gradio callbacks in `try/except` returns a readable error string to the UI instead. Startup validation in `config.py` catches missing env vars before any API call is made. The empty-chunks guard prevents a confusing Chroma error when someone uploads a scanned PDF.

---

### Task 4: Add startup validation to `config.py`

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Add validation after the `os.getenv` lines**

  After the two `os.getenv(...)` lines, add:

  ```python
  # Fail fast at import time — a clear error here is better than a cryptic
  # authentication failure on the first API call.
  _missing = [v for v in ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL") if not os.getenv(v)]
  if _missing:
      raise EnvironmentError(
          f"Required environment variables not set: {', '.join(_missing)}\n"
          "Copy .env.example to .env and fill in the values."
      )
  ```

- [ ] **Step 2: Verify — clear error when env vars are missing**

  Temporarily rename `.env` to `.env.bak`:
  ```bash
  cd notebooks/02-build-rag-applications-get-started/module-02/02-lab-construct-a-qa-bot-that-leverages-the-langchain-and-llm-to-answer-questions-from-loaded-document && \
  mv ../../../../../../.env ../../../../../../.env.bak && \
  python -c "import config" 2>&1; mv ../../../../../../.env.bak ../../../../../../.env
  ```
  Expected output contains:
  ```
  EnvironmentError: Required environment variables not set: OPENROUTER_API_KEY, OPENROUTER_BASE_URL
  Copy .env.example to .env and fill in the values.
  ```

---

### Task 5: Add error handling to Gradio callbacks

**Files:**
- Modify: `01-qabot.py`

- [ ] **Step 1: Wrap `build_retriever` in try/except**

  Replace the current `build_retriever` body with:

  ```python
  def build_retriever(file):
      if file is None:
          return None, "No file uploaded."
      logger.info("Building retriever for: %s", file)
      try:
          splits = document_loader(file)
          chunks = text_splitter(splits)
          logger.info("Split into %d chunks; embedding now...", len(chunks))

          # Scanned/image-only PDFs produce no extractable text — warn the user
          # rather than passing an empty list to Chroma, which raises unhelpfully.
          if not chunks:
              return None, "No text could be extracted from this PDF. Is it a scanned image?"

          vectordb = vector_database(chunks, EMBEDDING_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, EMBEDDING_DIMENSIONS)
          retriever_obj = vectordb.as_retriever()
          logger.info("Retriever ready")
          return retriever_obj, f"Document indexed: {len(chunks)} chunks ready."
      except Exception as exc:
          logger.exception("Failed to build retriever")
          return None, f"Failed to process document: {exc}"
  ```

- [ ] **Step 2: Wrap `answer_question` in try/except**

  Replace the current `answer_question` body with:

  ```python
  def answer_question(query, retriever_obj):
      if retriever_obj is None:
          return "Please process a document first."
      if not query or not query.strip():
          return "Please enter a question."
      logger.info("Answering query: %r", query)
      try:
          llm = llm_model(
              model_name=LLM_MODEL,
              model_api_key=OPENROUTER_API_KEY,
              model_base_url=OPENROUTER_BASE_URL,
              model_temperature=LLM_TEMPERATURE,
              model_maxtokens=LLM_MAX_TOKENS,
          )
          chain = (
              {"context": retriever_obj | format_docs, "question": RunnablePassthrough()}
              | RAG_PROMPT
              | llm
              | StrOutputParser()
          )
          answer = chain.invoke(query)
          logger.info("Answer length: %d chars", len(answer))
          return answer
      except Exception as exc:
          logger.exception("Failed to answer question")
          return f"Failed to generate an answer: {exc}"
  ```

- [ ] **Step 3: Start the app and verify all error paths**

  ```bash
  cd notebooks/02-build-rag-applications-get-started/module-02/02-lab-construct-a-qa-bot-that-leverages-the-langchain-and-llm-to-answer-questions-from-loaded-document && \
  python 01-qabot.py
  ```
  Open `http://localhost:7860` and confirm:

  | Action | Expected result in UI box |
  |--------|--------------------------|
  | Click **Process Document** without a file | `"No file uploaded."` |
  | Click **Ask** without processing first | `"Please process a document first."` |
  | Click **Ask** with empty question | `"Please enter a question."` |
  | Upload valid PDF → **Process Document** | `"Document indexed: N chunks ready."` |
  | Ask a question in the document | Coherent answer |
  | Ask a question not in the document | Response contains `"I don't know based on the provided document"` |

- [ ] **Step 4: Commit Batch 2**

  ```bash
  git add \
    notebooks/02-build-rag-applications-get-started/module-02/02-lab-construct-a-qa-bot-that-leverages-the-langchain-and-llm-to-answer-questions-from-loaded-document/config.py \
    notebooks/02-build-rag-applications-get-started/module-02/02-lab-construct-a-qa-bot-that-leverages-the-langchain-and-llm-to-answer-questions-from-loaded-document/01-qabot.py
  git commit -m "fix(qabot): startup env validation, user-friendly error handling in Gradio callbacks"
  ```

---

## Batch 3 — Code Organisation

**Closes:** issues #11–12  
**Breaking changes:** None — same public API, same behaviour

### What and why

Creating a new `ChatOpenRouter` object on every question wastes time and scatters configuration. Making it a module-level constant means it is built once, from values already validated at startup. Merging `embedding_model` into `vector_database` removes an unnecessary indirection whose only effect is duplicating the parameter list.

---

### Task 6: Cache the LLM as a module-level constant

**Files:**
- Modify: `01-qabot.py`

- [ ] **Step 1: Add a module-level `LLM` constant**

  After the `RAG_PROMPT` line, add:

  ```python
  # LLM is stateless — built once at module load rather than once per question.
  LLM = llm_model(
      model_name=LLM_MODEL,
      model_api_key=OPENROUTER_API_KEY,
      model_base_url=OPENROUTER_BASE_URL,
      model_temperature=LLM_TEMPERATURE,
      model_maxtokens=LLM_MAX_TOKENS,
  )
  ```

- [ ] **Step 2: Remove the per-call instantiation in `answer_question`**

  Remove the local `llm = llm_model(...)` block from `answer_question` and update the chain to reference `LLM`:

  ```python
  chain = (
      {"context": retriever_obj | format_docs, "question": RunnablePassthrough()}
      | RAG_PROMPT
      | LLM
      | StrOutputParser()
  )
  ```

- [ ] **Step 3: Verify `llm_model` is called exactly once**

  ```bash
  grep -c "llm_model(" \
    notebooks/02-build-rag-applications-get-started/module-02/02-lab-construct-a-qa-bot-that-leverages-the-langchain-and-llm-to-answer-questions-from-loaded-document/01-qabot.py
  ```
  Expected: `2` — function definition + the single module-level `LLM = llm_model(...)` call.

---

### Task 7: Merge `embedding_model` and `vector_database` into `build_vector_store`

**Files:**
- Modify: `01-qabot.py`

- [ ] **Step 1: Replace both functions with `build_vector_store`**

  Delete `embedding_model` and `vector_database` entirely and replace with:

  ```python
  def build_vector_store(chunks):
      # Embeds chunks into an in-memory Chroma store. The store is ephemeral —
      # it lives only for this session and is rebuilt on each new document upload.
      logger.info("Embedding %d chunks (model=%s, dim=%d)", len(chunks), EMBEDDING_MODEL, EMBEDDING_DIMENSIONS)
      embeddings = OpenAIEmbeddings(
          model=EMBEDDING_MODEL,
          openai_api_key=OPENROUTER_API_KEY,
          openai_api_base=OPENROUTER_BASE_URL,
          dimensions=EMBEDDING_DIMENSIONS,
      )
      return Chroma.from_documents(chunks, embeddings)
  ```

- [ ] **Step 2: Update the call site in `build_retriever`**

  ```python
  # Before:
  vectordb = vector_database(chunks, EMBEDDING_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, EMBEDDING_DIMENSIONS)

  # After:
  vectordb = build_vector_store(chunks)
  ```

- [ ] **Step 3: Verify old function names are gone**

  ```bash
  grep -n "embedding_model\|vector_database" \
    notebooks/02-build-rag-applications-get-started/module-02/02-lab-construct-a-qa-bot-that-leverages-the-langchain-and-llm-to-answer-questions-from-loaded-document/01-qabot.py
  ```
  Expected: no output

- [ ] **Step 4: Run the full happy path**

  Start the app, upload a PDF, process it, ask a question. Confirm an answer is returned and the terminal shows the embedding log line.

- [ ] **Step 5: Commit Batch 3**

  ```bash
  git add \
    notebooks/02-build-rag-applications-get-started/module-02/02-lab-construct-a-qa-bot-that-leverages-the-langchain-and-llm-to-answer-questions-from-loaded-document/01-qabot.py
  git commit -m "refactor(qabot): cache LLM at module level, merge embedding+vector into build_vector_store"
  ```

---

## Batch 4 — Comments and Documentation

**Closes:** issues #13–15  
**Breaking changes:** None

### What and why

Comments that name the thing they describe add no information. `## Document loading` above `def document_loader` is pure noise. The `##` double-hash style is unconventional Python. Good comments explain non-obvious decisions — *why* `1024` dimensions, *why* the LCEL chain is structured that way.

---

### Task 8: Rewrite inline comments in `01-qabot.py`

**Files:**
- Modify: `01-qabot.py`

- [ ] **Step 1: Remove the four redundant section markers**

  Delete these lines entirely — the function names already convey what they do:
  ```python
  ## Document loading
  ## Text splitting
  ## Embedding model   ← already removed by Task 7
  ## Vector db         ← already removed by Task 7
  ```

- [ ] **Step 2: Improve the two useful markers**

  ```python
  # Before:
  ## Build retriever — indexing step, runs once per document

  # After (single hash, explains why):
  # Indexing step — runs once per document upload; result cached in gr.State so
  # subsequent questions reuse the retriever without re-embedding.
  ```

  ```python
  # Before:
  ## Answer question using cached retriever

  # After:
  # Query step — receives the cached retriever from gr.State and runs the LCEL chain.
  ```

- [ ] **Step 3: Remove the two obvious UI comments**

  Delete:
  ```python
  # Create Gradio Blocks interface   ← obvious from `gr.Blocks`
  # Launch the app                   ← obvious from `.launch(...)`
  ```

- [ ] **Step 4: Add an explanatory comment above the LCEL chain**

  ```python
  # LCEL chain — reads left to right:
  # retriever fetches the most relevant chunks → format_docs joins them into one
  # context string → RAG_PROMPT fills {context} and {question} → LLM generates
  # the answer → StrOutputParser extracts the plain string from the message object.
  chain = (
      {"context": retriever_obj | format_docs, "question": RunnablePassthrough()}
      | RAG_PROMPT
      | LLM
      | StrOutputParser()
  )
  ```

- [ ] **Step 5: Verify no `##` markers remain**

  ```bash
  grep -n "^##" \
    notebooks/02-build-rag-applications-get-started/module-02/02-lab-construct-a-qa-bot-that-leverages-the-langchain-and-llm-to-answer-questions-from-loaded-document/01-qabot.py
  ```
  Expected: no output

---

### Task 9: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the Configuration section to mention `config.py`**

  Replace the existing intro sentence of the Configuration section with:

  ```markdown
  Non-secret configuration (model names, tuning parameters, Gradio settings, prompt template) lives in `config.py`. Secrets are the only things that belong in `.env`.

  Copy `.env.example` to `.env` and fill in the two values:
  ```

  Update the configuration table to split into two parts:

  ```markdown
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
  ```

- [ ] **Step 2: Add a Troubleshooting section at the end of README.md**

  ```markdown
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
  ```

- [ ] **Step 3: Commit Batch 4**

  ```bash
  git add \
    notebooks/02-build-rag-applications-get-started/module-02/02-lab-construct-a-qa-bot-that-leverages-the-langchain-and-llm-to-answer-questions-from-loaded-document/01-qabot.py \
    notebooks/02-build-rag-applications-get-started/module-02/02-lab-construct-a-qa-bot-that-leverages-the-langchain-and-llm-to-answer-questions-from-loaded-document/README.md \
    notebooks/02-build-rag-applications-get-started/module-02/02-lab-construct-a-qa-bot-that-leverages-the-langchain-and-llm-to-answer-questions-from-loaded-document/PLAN.md
  git commit -m "docs(qabot): rewrite why-comments, update README with config.py split and troubleshooting"
  ```

---

## Final Structure of `01-qabot.py`

```
imports (from config + langchain + gradio)
logging setup
RAG_PROMPT  (ChatPromptTemplate built from config.RAG_PROMPT_TEMPLATE)
LLM         (ChatOpenRouter, built once at module load)
─── pipeline functions ──────────────────────────────
document_loader(file_path)        → list[Document]
text_splitter(docs)               → list[Document] (chunked)
build_vector_store(chunks)        → Chroma
format_docs(docs)                 → str
─── Gradio callbacks ────────────────────────────────
build_retriever(file)             → (retriever | None, status_str)
answer_question(query, retriever) → answer_str
─── UI + launch ─────────────────────────────────────
gr.Blocks definition + event wiring
rag_application.launch(...)
```

## Batch Test Checklist

| Batch | Pass criteria |
|-------|--------------|
| 1 | `python -c "import config"` succeeds; `grep os.getenv 01-qabot.py` returns no output; `.env.example` has exactly 2 non-comment lines |
| 2 | Missing env raises clear `EnvironmentError`; all 6 UI error paths return readable strings, not tracebacks |
| 3 | `grep -c "llm_model(" 01-qabot.py` → `2`; `grep "embedding_model\|vector_database" 01-qabot.py` → no output; full PDF→answer flow works |
| 4 | `grep "^##" 01-qabot.py` → no output; LCEL chain has explanatory comment; README has Troubleshooting section |
