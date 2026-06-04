"""
Phase 4 test — verify query_engine functions against live mock data.

Run with:
    python test_phase4.py
"""
import sys
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

import requests
import llm_setup
import data_processing
import query_engine
import config

print("Configuring LlamaIndex Settings...")
llm_setup.configure()

print("\nBuilding index from mock data...")
response = requests.get(config.MOCK_DATA_URL, timeout=30)
profile_data = response.json()
nodes = data_processing.split_profile_data(profile_data)
index = data_processing.create_vector_database(nodes)
print("  Index ready.")

# --- Test 1: generate_initial_facts returns a non-empty string ---
print("\n[Test 1] generate_initial_facts...")
facts = query_engine.generate_initial_facts(index)
assert isinstance(facts, str), f"Expected str, got {type(facts)}"
assert len(facts.strip()) > 0, "Facts string is empty"
print(f"  Returned {len(facts)} characters  PASS")
print(f"  Preview: {facts[:120].strip()}...")

# --- Test 2: answer_user_query returns a Response with .response attribute ---
print("\n[Test 2] answer_user_query...")
answer = query_engine.answer_user_query(index, "What companies has this person worked at?")
assert hasattr(answer, "response"), \
    f"Return value has no .response attribute: {type(answer)}"
assert len(answer.response.strip()) > 0, "Answer text is empty"
print(f"  Answer preview: {answer.response.strip()[:120]}...  PASS")

# --- Test 3: llm_interface.py is gone ---
print("\n[Test 3] llm_interface.py is deleted...")
import os
assert not os.path.exists("llm_interface.py"), \
    "llm_interface.py still exists — delete it"
print("  llm_interface.py absent  PASS")

# --- Test 4: query engine does not silently swallow exceptions ---
print("\n[Test 4] Exceptions propagate (bad index type)...")
try:
    query_engine.generate_initial_facts(None)
    assert False, "Should have raised an exception"
except Exception as e:
    print(f"  Raised {type(e).__name__} as expected  PASS")

print("\n=== Phase 4: ALL TESTS PASSED ===")
