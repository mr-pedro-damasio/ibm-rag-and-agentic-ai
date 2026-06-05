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

# Gradio server
GRADIO_SERVER_NAME = "localhost"
GRADIO_SERVER_PORT = 7860
GRADIO_SHARE = False  # set to True to create a public Gradio tunnel (use with caution)
