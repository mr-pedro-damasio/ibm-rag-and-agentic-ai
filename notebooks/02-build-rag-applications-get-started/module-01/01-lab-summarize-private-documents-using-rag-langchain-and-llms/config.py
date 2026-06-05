import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")

_missing = [v for v in ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL") if not os.getenv(v)]
if _missing:
    raise EnvironmentError(
        f"Required environment variables not set: {', '.join(_missing)}\n"
        "Copy .env.example to .env and fill in the values."
    )

LLM_MODEL = "openai/gpt-4o-mini"
LLM_TEMPERATURE = 0.5
LLM_MAX_TOKENS = 512
LLM_MAX_COMPLETION_TOKENS = 256

EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1024

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
