"""
Phase 1 test — verify that llm_setup.configure() correctly initialises
the LlamaIndex Settings singleton for both LLM and embedding model.

Run with:
    python test_phase1.py
"""
import sys
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

import llm_setup
from llama_index.core import Settings

print("Configuring LlamaIndex Settings via llm_setup.configure()...")
llm_setup.configure()

# --- Test 1: Settings.llm is set and responds ---
print("\n[Test 1] LLM basic completion...")
response = Settings.llm.complete("Reply with exactly one word: hello")
assert response.text.strip(), "LLM response should not be empty"
print(f"  LLM response: '{response.text.strip()}'  PASS")

# --- Test 2: Settings.embed_model is set and returns correct dimensions ---
print("\n[Test 2] Embedding dimensions...")
embedding = Settings.embed_model.get_text_embedding("Hello world")
assert len(embedding) == 1024, f"Expected 1024 dimensions, got {len(embedding)}"
print(f"  Embedding dimensions: {len(embedding)}  PASS")

# --- Test 3: model_name override works ---
print("\n[Test 3] model_name override...")
import config
llm_setup.configure(model_name="openai/gpt-4o-mini")
assert Settings.llm.model == "openai/gpt-4o-mini", \
    f"Expected 'openai/gpt-4o-mini', got '{Settings.llm.model}'"
print(f"  Model after override: '{Settings.llm.model}'  PASS")

# Restore default
llm_setup.configure()

print("\n=== Phase 1: ALL TESTS PASSED ===")
