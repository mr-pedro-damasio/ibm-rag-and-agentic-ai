import os
from dotenv import load_dotenv

load_dotenv()

# --- Gradio server ---
GRADIO_SERVER_NAME = "0.0.0.0"  # 0.0.0.0 required for container / Codespace port forwarding
GRADIO_SERVER_PORT = 7860
GRADIO_SHARE = False

# --- OpenRouter credentials ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")

# --- LLM model defaults ---
LLM_MODEL = "openai/gpt-4o-mini"
LLM_TEMPERATURE = 0.5
LLM_MAX_TOKENS = 256
LLM_MAX_COMPLETION_TOKENS = 128
