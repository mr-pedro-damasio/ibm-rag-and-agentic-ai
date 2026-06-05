# AI Icebreaker Bot

A RAG-powered chatbot that ingests a LinkedIn profile, vectorizes it with LlamaIndex, and answers questions about the person using an LLM routed through OpenRouter.

---

## Architecture

The pipeline has two distinct phases: **indexing** (runs once at startup) and **querying** (runs on every user question). Both phases share the same LLM and embedding model, built once by `llm_setup.configure()` and passed explicitly through the call chain.

### High-level flow

```
┌─────────────────────────────────────────────────────────────────┐
│  STARTUP                                                        │
│                                                                 │
│  main.py (argparse)                                             │
│      │                                                          │
│      ▼                                                          │
│  setup = llm_setup.configure()                                  │
│      ├── setup.llm       = OpenAILike    ──► OpenRouter LLM     │
│      ├── setup.embed_model = OpenAIEmbedding ──► OpenRouter     │
│      └── Settings.chunk_size = 500  (safety-net write)          │
└─────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  INDEXING PHASE                                                 │
│                                                                 │
│  profile_extraction.py                                          │
│      └── requests.get() ──► ProxyCurl API  (live)              │
│                        └──► S3 mock URL    (--mock)             │
│      returns dict (cleaned JSON)                                │
│                │                                                │
│                ▼                                                │
│  data_processing.split_profile_data()                           │
│      └── groups keys into semantic buckets                      │
│          └── json.dumps each bucket → Document(metadata=section)│
│              └── SentenceSplitter → [TextNode, TextNode, ...]   │
│                                                                 │
│  data_processing.create_vector_database()                       │
│      └── VectorStoreIndex(nodes)                                │
│          ├── OpenAIEmbedding.get_text_embedding(node.text)      │
│          │       └──► POST /embeddings → 1024-dim float vector  │
│          ├── SimpleDocumentStore  (node_id → TextNode)          │
│          └── SimpleVectorStore    (node_id → embedding vector)  │
└─────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  QUERY PHASE  (repeated per user question)                      │
│                                                                 │
│  query_engine.generate_initial_facts() / answer_user_query()   │
│      └── VectorStoreIndex.as_query_engine()                     │
│              └── RetrieverQueryEngine                           │
│                  │                                              │
│                  ├── RETRIEVAL                                  │
│                  │   ├── embed query → POST /embeddings         │
│                  │   └── cosine similarity vs SimpleVectorStore │
│                  │       └── top-k TextNode chunks              │
│                  │                                              │
│                  └── SYNTHESIS                                  │
│                      ├── PromptTemplate fills {context_str}     │
│                      │                       + {query_str}      │
│                      └── OpenAILike → POST /chat/completions    │
│                              └── response.response (str)        │
└─────────────────────────────────────────────────────────────────┘
```

---

### Step-by-step event flow

#### 1. Startup — `main.py`

`argparse` parses the CLI flags (`--mock`, `--url`, `--model`, `--api-key`, `--log-level`). `logging.basicConfig()` configures the root logger to write structured output to stderr.

#### 2. Settings bootstrap — `llm_setup.configure(model_name)`

This is the first substantive call and must complete before anything else runs. It populates the LlamaIndex `Settings` singleton (from `llama_index.core`):

`configure()` builds two objects and returns them as an `LLMSetup` dataclass. It also writes them to the LlamaIndex `Settings` singleton as a safety net for any internal LlamaIndex components that bypass explicit arguments.

- **`llm`** — an `OpenAILike` instance (`llama_index.llms.openai_like`). Unlike the `OpenAI` adapter (which is hard-coded to OpenAI's API), `OpenAILike` accepts any OpenAI-protocol endpoint. It is pointed at `https://openrouter.ai/api/v1` with `is_chat_model=True`, so it uses the `/chat/completions` endpoint. The model name is passed through as-is (e.g. `"openai/gpt-4o-mini"`) — no prefix manipulation needed.

- **`embed_model`** — an `OpenAIEmbedding` instance (`llama_index.embeddings.openai`). Also pointed at OpenRouter's base URL. Because `OpenAIEmbedding` forwards the model name directly to the `/embeddings` endpoint and OpenRouter's embeddings endpoint does not accept provider prefixes, the `"openai/"` prefix is stripped here (e.g. `"text-embedding-3-small"`). `dimensions=1024` requests a compressed output rather than the default 1536.

- **`Settings.chunk_size`** — set to 500 tokens; used automatically by `SentenceSplitter` when it reads from `Settings`.

The returned `LLMSetup(llm, embed_model)` is then passed explicitly to `create_vector_database()`, `generate_initial_facts()`, and `answer_user_query()` so that each function's model dependency is visible in its signature.

#### 3. Profile ingestion — `profile_extraction.extract_linkedin_profile()`

Uses the `requests` library to fetch the LinkedIn profile as JSON:

- **Mock mode**: `GET https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/...` — a static JSON file hosted on IBM Cloud Object Storage.
- **Live mode**: `GET https://nubela.co/proxycurl/api/v2/linkedin` with `Authorization: Bearer <key>` and the target LinkedIn URL as a query parameter. The ProxyCurl API scrapes LinkedIn and returns a structured JSON profile.

Either path returns a raw `dict`. The function then cleans it in-place: it drops keys whose value is `[]`, `""`, or `None`; removes the `people_also_viewed` and `certifications` keys entirely; and strips `profile_pic_url` from each entry in `groups`.

#### 4. Document construction — `data_processing.split_profile_data()`

The cleaned `dict` is split into semantic sections. Each section is serialised with `json.dumps(content, indent=2)` and wrapped in a `Document` object (`llama_index.core.Document`) with a `metadata={"section": <tag>}` dict attached.

Sections produced:

| `section` tag | Source keys | Granularity |
|---|---|---|
| `header` | `first_name`, `last_name`, `full_name`, `headline`, `summary`, `occupation`, `country`, `city`, `state` | One document |
| `experience` | `experiences[i]` | One document per job entry |
| `education` | `education[i]` | One document per entry |
| `skills` | `skills`, `languages` | One document (combined) |
| `activity` | `groups`, `recommendations`, `accomplishment_*`, `volunteer_work` | One document (combined) |
| `other` | Any remaining keys | One document |

The list of `Document` objects is then fed to `SentenceSplitter(chunk_size=500)` (`llama_index.core.node_parser`). The splitter tokenises each document and breaks it into `TextNode` objects at sentence boundaries, keeping each node within the 500-token limit. Every `TextNode` inherits the `section` metadata from its parent `Document` and is assigned a UUID as its `node_id`.

#### 5. Indexing — `data_processing.create_vector_database()`

`VectorStoreIndex(nodes=nodes, embed_model=embed_model)` (`llama_index.core`) iterates over every `TextNode` and calls `embed_model.get_text_embedding(node.text)`. Each call makes a POST request to `https://openrouter.ai/api/v1/embeddings`, which routes to the OpenAI `text-embedding-3-small` model. The response is a 1024-dimensional `list[float]`.

The index stores two things in a `StorageContext`:
- **`SimpleDocumentStore`** — maps `node_id → TextNode` (full text + metadata).
- **`SimpleVectorStore`** — maps `node_id → embedding vector` (stored in `SimpleVectorStoreData.embedding_dict`). Both stores are held in-memory; there is no external vector database.

`data_processing.verify_embeddings()` confirms that the set of `node_id` keys in the docstore exactly matches those in the embedding dict, catching any nodes that failed to embed.

#### 6. Initial facts — `query_engine.generate_initial_facts()`

`index.as_query_engine(llm=llm, similarity_top_k=5, text_qa_template=PromptTemplate(...))` constructs a `RetrieverQueryEngine` backed by a `VectorIndexRetriever`. The engine's `query()` method runs two sub-steps:

**Retrieval:** The query string `"Provide three interesting facts..."` is embedded (another POST to `/embeddings`). The resulting vector is compared against every embedding in `SimpleVectorStore` using cosine similarity. The top-5 `TextNode` chunks are returned as `NodeWithScore` objects.

**Synthesis:** The retrieved chunks are concatenated into a `context_str`. The `PromptTemplate` (`INITIAL_FACTS_TEMPLATE`) substitutes `{context_str}` with the retrieved text. The filled prompt is sent to the `llm` as a POST to `https://openrouter.ai/api/v1/chat/completions`. The `response.response` attribute holds the plain-text answer string, which is returned and printed.

#### 7. Interactive chatbot — `main.py chatbot_interface()`

A `while True` loop reads user input via `input("You: ")`. Each non-exit input is passed to `query_engine.answer_user_query(index, user_query)`, which runs the same retrieval + synthesis pipeline as above but uses `USER_QUESTION_TEMPLATE` (which includes a `{query_str}` slot for the specific question). The function returns a LlamaIndex `Response` object; `main.py` accesses `.response` for the text and prints it. The loop exits when the user types `exit`, `quit`, or `bye`.

---

### Libraries and backends

| Library | Package | Role |
|---|---|---|
| LlamaIndex core | `llama-index-core` | `Settings`, `Document`, `VectorStoreIndex`, `SentenceSplitter`, `PromptTemplate`, `RetrieverQueryEngine`, `SimpleVectorStore`, `SimpleDocumentStore` |
| LlamaIndex LLM adapter | `llama-index-llms-openai-like` | `OpenAILike` — OpenAI-protocol adapter for third-party providers |
| LlamaIndex embedding adapter | `llama-index-embeddings-openai` | `OpenAIEmbedding` — calls `/embeddings` endpoint |
| OpenRouter | API gateway | Routes LLM and embedding requests to the actual model provider (OpenAI, IBM, etc.) |
| ProxyCurl | External API | Scrapes and returns LinkedIn profiles as structured JSON (live mode only) |
| requests | `requests` | HTTP client for profile fetching and (indirectly) embedding/LLM calls |
| python-dotenv | `python-dotenv` | Loads `OPENROUTER_API_KEY` and `PROXY_CURL_API_KEY` from `.env` |

---

## File Reference

| File | Role |
|---|---|
| `main.py` | CLI entry point. Parses args, calls `llm_setup.configure()`, runs the pipeline. |
| `llm_setup.py` | Builds `OpenAILike` (LLM) and `OpenAIEmbedding` (embed model), writes them to `Settings` as a safety net, and returns them as an `LLMSetup` dataclass. Accepts an optional `model_name` override for the `--model` flag. |
| `config.py` | All constants: API keys (from env), model names, chunk size, prompt templates. |
| `profile_extraction.py` | Fetches LinkedIn JSON via ProxyCurl or a mock S3 URL. Cleans empty fields. |
| `data_processing.py` | Splits the profile into one `Document` per semantic section, builds a `VectorStoreIndex`. |
| `query_engine.py` | Generates initial facts and answers free-form questions using the index. |

---

## Configuration

Copy `.env.example` to `.env` and set:

```
OPENROUTER_API_KEY=<your key>
PROXY_CURL_API_KEY=<your key>   # only needed for real LinkedIn URLs
```

Key constants in `config.py`:

| Constant | Default | Purpose |
|---|---|---|
| `QUERY_MODEL` | `openai/gpt-4o-mini` | LLM for facts and Q&A |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Embedding model |
| `EMBEDDING_MODEL_DIMENSIONS` | `1024` | Vector dimensions |
| `QUERY_MODEL_TEMPERATURE` | `0.0` | LLM temperature |
| `QUERY_MODEL_MAX_TOKENS` | `2048` | Max LLM output tokens |
| `QUERY_MODEL_TOP_K` | `5` | Retrieved nodes per query |
| `CHUNK_SIZE` | `500` | Token chunk size for splitting |

---

## Running

**With mock data (no API keys required beyond OpenRouter):**

```bash
python main.py --mock
```

**With a real LinkedIn URL:**

```bash
python main.py --url https://www.linkedin.com/in/someone/ --api-key <proxycurl-key>
```

**Override the LLM at runtime:**

```bash
python main.py --mock --model ibm/granite-3-2-8b-instruct
```

**Adjust log verbosity:**

```bash
python main.py --mock --log-level DEBUG
```

Once the bot loads the profile it prints three interesting facts, then enters an interactive Q&A loop. Type `exit`, `quit`, or `bye` to stop.

---

## Semantic Chunking

The LinkedIn JSON is split into semantic sections rather than a single raw dump. Each section becomes its own `Document` with a `section` metadata tag:

| Section tag | Content |
|---|---|
| `header` | Name, headline, summary, location, occupation |
| `experience` | One document per job entry |
| `education` | One document per education entry |
| `skills` | Skills and languages combined |
| `activity` | Groups, recommendations, accomplishments |
| `other` | Any remaining top-level keys |

This keeps chunk boundaries aligned with meaning, improving retrieval precision.

