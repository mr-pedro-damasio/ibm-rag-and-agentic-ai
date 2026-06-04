"""
Phase 2 test — verify config.py has the expected constants and no dead ones.

Run with:
    python test_phase2.py
"""
import config

REQUIRED = [
    "MOCK_DATA_URL", "LINKEDIN_API_ENDPOINT", "PROXYCURL_API_KEY",
    "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL",
    "EMBEDDING_MODEL", "EMBEDDING_MODEL_DIMENSIONS",
    "QUERY_MODEL", "QUERY_MODEL_TEMPERATURE", "QUERY_MODEL_MAX_TOKENS",
    "QUERY_MODEL_TOP_K", "QUERY_MODEL_TOP_P",
    "CHUNK_SIZE",
    "INITIAL_FACTS_TEMPLATE", "USER_QUESTION_TEMPLATE",
]

MUST_NOT_EXIST = [
    "EMBEDDING_MODEL_TEMPERATURE",       # meaningless for embeddings
    "QUERY_MODEL_MAX_COMPLETION_TOKENS", # never applied to any constructor
]

print("[Test 1] Required constants present...")
for key in REQUIRED:
    assert hasattr(config, key), f"  MISSING: config.{key}"
    print(f"  config.{key}  PASS")

print("\n[Test 2] Dead constants removed...")
for key in MUST_NOT_EXIST:
    assert not hasattr(config, key), \
        f"  config.{key} still present — should have been removed"
    print(f"  config.{key} absent  PASS")

print("\n[Test 3] OpenRouter credentials are loaded from environment...")
assert config.OPENROUTER_API_KEY, \
    "OPENROUTER_API_KEY is empty — check your .env file"
print("  OPENROUTER_API_KEY loaded  PASS")

print("\n=== Phase 2: ALL TESTS PASSED ===")
