# PLAN — Notebook Refactoring

## Architecture Reference

The module-02 labs establish the pattern:

```
lab-directory/
├── config.py          # Env vars, model/embedding/chunking settings, prompt template
├── llm_setup.py       # llm_model(), llm_response() — factory functions using config
└── lab-script.py      # imports from config, llm_setup; overrides only what differs
```

**Note on the two config.py variants in module-02:**
- `01-lab/config.py` — no validation, has `LLM_MAX_COMPLETION_TOKENS`
- `02-lab/config.py` — has import-time validation, no `LLM_MAX_COMPLETION_TOKENS`, has `RAG_PROMPT_TEMPLATE`

This notebook's `config.py` will merge the best of both: validation (from 02-lab) + `LLM_MAX_COMPLETION_TOKENS` (from 01-lab, since the notebook uses it) + RAG-specific settings.

---

## Batch 0: Extract Shared Modules (config.py + llm_setup.py)

**Goal**: Create `config.py` and `llm_setup.py` following the module-02 pattern. The notebook imports from these instead of defining `llm_model()` / `llm_model_response()` inline.

### 0.1 Create `config.py`

Modeled on `02-lab-construct-a-qa-bot/config.py` with additions from `01-lab-llm-chat/config.py`.

**Contents:**
- `load_dotenv()`, read `OPENROUTER_API_KEY` + `OPENROUTER_BASE_URL` from env
- Import-time validation — `raise EnvironmentError(...)` if either is missing  
  (clear message: "Copy .env.example to .env and fill in the values.")
- `LLM_MODEL = "openai/gpt-4o-mini"` — model for RAG (not the default granite)
- `LLM_TEMPERATURE = 0.5`, `LLM_MAX_TOKENS = 256`, `LLM_MAX_COMPLETION_TOKENS = 128`
- `EMBEDDING_MODEL = "openai/text-embedding-3-small"`, `EMBEDDING_DIMENSIONS = 1024`
- `CHUNK_SIZE = 1000`, `CHUNK_OVERLAP = 0` (matches current notebook behavior)

**Test**: `python -c "from config import *"` succeeds when `.env` is present, raises clear error when absent.

### 0.2 Create `llm_setup.py`

Modeled on `01-lab-llm-chat/llm_setup.py`.

**Contents:**
- Import `config` module
- `llm_model(params: dict | None = None) -> ChatOpenRouter` — same signature as module-02
- `llm_response(prompt_text: str, params: dict | None = None) -> str` — convenience wrapper

**Test**: `from llm_setup import llm_model, llm_response` succeeds.

### 0.3 Update notebook imports

**Problem**: Notebook hardcodes `llm_model()` and `llm_model_response()` inline.

**Fix**: Add a cell that adds the notebook directory to `sys.path`, then:
```python
from config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL,
    EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, CHUNK_SIZE, CHUNK_OVERLAP,
)
from llm_setup import llm_model, llm_response
```

Remove the inline `llm_model()` / `llm_model_response()` function definitions.

**Test**: `llm_model()` and `llm_response()` work identically to before.

### 0.4 Reduce params dict redundancy

**Problem**: Module-02 scripts override only what differs from config defaults, e.g.:
```python
params = {"temperature": 0.8, "max_tokens": 1024, "max_completion_tokens": 512}
```
But the notebook still passes `api_key` and `base_url` redundantly.

**Fix**: Follow the module-02 pattern — override dicts include only the fields that differ.

**Test**: Model initializes with the same effective parameters.

---

## Batch 1: Fix Bugs & Dead Code

**Goal**: Deliver a notebook that runs correctly with no logic bugs. Each cell must produce correct, observable output.

### 1.1 Fix `qa_agent()` chat_history bug (CRITICAL)

**Problem** (line 629): The `qa_agent()` function initializes `qa_agent_chat_history = []` but passes `chat_history` (the outer-scope variable) to the chain invoke. This means the interactive agent loop never accumulates its own conversation history — every question after the first loses context.

**Fix**: Change `"chat_history": chat_history` → `"chat_history": qa_agent_chat_history` in the `qa_agent()` function.

**Test**: Run `qa_agent()`. Ask "what is the mobile policy?", then "What I cannot do in it?". The second answer must reference mobile policy (not internet/email policy).

### 1.2 Remove duplicate `qa_agent()` reimplementation (DEAD CODE)

**Problem**: The `qa_agent()` function (lines 516–641) duplicates the exact same chain-building logic already present in the "Make the conversation have memory" section (lines 331–426). This is copy-paste dead code.

**Fix**: Replace the entire `qa_agent()` function body with a simple interactive loop that reuses the existing `qa_with_memory` chain and `chat_history` list already built above it.

**Test**: The interactive agent produces correct answers, the duplicate ~80 lines are gone, and results match Batch 1.1.

### 1.3 Remove redundant `retriever = chroma_doc_search.as_retriever()` at line 229

**Problem**: The retriever is reassigned at line 334 for the memory section. Line 229 creates it for the basic LCEL chain, which works fine inline.

**Fix**: Keep line 229 (needed for the basic chain), remove the reassignment at line 334 (it creates a second identical retriever).

**Test**: The basic chain at line 237 and the memory chain at line 420 both still work correctly.

---

## Batch 2: Remove Dead Imports & Redundancy

**Goal**: Clean up imports and eliminate unnecessary parameter duplication.

### 2.1 Remove `wget` dependency, use `urllib.request`

**Problem**: `from wget import download` adds an external dependency for a single download call. Python's `urllib.request.urlretrieve` does the same job with zero dependencies.

**Fix**: Replace the `wget` import and `download()` call with `urllib.request.urlretrieve()`.

**Test**: The download cell runs and produces the same `companyPolicies.txt` file.

---

## Batch 3: Comment Quality Pass

**Goal**: Pare down excessively verbose comments that restate the obvious, while preserving educational value.

### 3.1 Trim paragraph-length obvious comments

**Problem**: Many comments explain what the next line of code does in paragraph form (e.g., `# Print the answer to the second question` before `print()`). Jupyter markdown cells already provide the narrative.

**Fix**: Remove comments that merely restate the next line of code. Keep comments that explain *why* something is done, not *what* is done.

**Test**: No functional change. Notebook remains pedagogically clear.

### 3.2 Normalize comment style

**Problem**: The memory section has inline block comments (lines 331–426) that look like they were generated by an AI assistant — 5-6 lines of prose per code block. The rest of the notebook has minimal comments.

**Fix**: Either remove the excessive block comments or reduce them to 1-2 line annotations.

**Test**: No functional change. Visual consistency restored.

---

## Batch 4: Structural Cleanup

**Goal**: Improve cell organization and consistency.

### 4.1 Reorder cells for logical flow

**Problem**: The notebook jumps from "Preprocessing" → "Load The Document" → "Embedding and storing" → "LLM Model Construction" → "Model" → back to "Dive Deeper" → "Prompt Template" → "Make the conversation have memory" → "Wrap it and make it an agent". The flow is reasonable but a bit fragmented.

**Fix**: No major reorder needed; consider adding a markdown summary cell before the interactive agent section that explains the final design.

**Test**: Still flows logically.

---

## Testing Procedure

After each batch:
1. Restart kernel (clear all state)
2. Run all cells sequentially
3. Verify outputs match expected behavior
4. Verify no import errors

For Batch 1 specifically, run the interactive agent with:
```
what is the mobile policy?
Can I eat in company vehicles?
What I cannot do in it?
```
→ Third answer should reference company vehicles, not internet/mobile policy.
