import os
from dotenv import load_dotenv
from typing import List

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")

def validate_environment_variables() -> None:
    """Validate all required environment variables are set."""
    required_vars = ["OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"]
    missing = [v for v in required_vars if not os.getenv(v)]
    
    if missing:
        error_msg = f"Required environment variables not set: {', '.join(missing)}"
        error_msg += "\nCopy .env.example to .env and fill in the values."
        error_msg += "\nMake sure to source the .env file or set the variables in your environment."
        raise EnvironmentError(error_msg)
    
    # Validate API key format (basic check)
    if OPENROUTER_API_KEY and len(OPENROUTER_API_KEY) < 20:
        print("Warning: API key seems unusually short. Please verify your OPENROUTER_API_KEY.")

# Run validation on import
validate_environment_variables()

LLM_MODEL = "openai/gpt-4o-mini"
LLM_TEMPERATURE = 0.5
LLM_MAX_TOKENS = 512
LLM_MAX_COMPLETION_TOKENS = 256

EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1024

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Add configuration validation
if CHUNK_SIZE <= CHUNK_OVERLAP:
    raise ValueError("CHUNK_SIZE must be greater than CHUNK_OVERLAP")

if CHUNK_SIZE <= 0 or CHUNK_OVERLAP < 0:
    raise ValueError("CHUNK_SIZE must be positive and CHUNK_OVERLAP must be non-negative")
