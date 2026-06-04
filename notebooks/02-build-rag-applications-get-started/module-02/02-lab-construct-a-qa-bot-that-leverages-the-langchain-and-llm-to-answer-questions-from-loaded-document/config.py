import os
import dotenv
dotenv.load_dotenv()

# Secrets — values come from .env, never hardcoded
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")

# Fail fast at import time — a clear error here is better than a cryptic
# authentication failure on the first API call.
_missing = [v for v in ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL") if not os.getenv(v)]
if _missing:
    raise EnvironmentError(
        f"Required environment variables not set: {', '.join(_missing)}\n"
        "Copy .env.example to .env and fill in the values."
    )

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
