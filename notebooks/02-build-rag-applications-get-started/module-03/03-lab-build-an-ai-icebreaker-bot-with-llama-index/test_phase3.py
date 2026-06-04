"""
Phase 3 test — verify semantic chunking, vector database creation,
and embedding verification using live mock data from S3.

Run with:
    python test_phase3.py
"""
import sys
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

import requests
import llm_setup
import data_processing
import config

print("Configuring LlamaIndex Settings...")
llm_setup.configure()

print("\nFetching mock LinkedIn profile from S3...")
response = requests.get(config.MOCK_DATA_URL, timeout=30)
assert response.status_code == 200, f"Failed to fetch mock data: {response.status_code}"
profile_data = response.json()
print(f"  Fetched profile: {profile_data.get('full_name', 'unknown')}")

# --- Test 1: split_profile_data returns nodes ---
print("\n[Test 1] Semantic chunking...")
nodes = data_processing.split_profile_data(profile_data)
assert len(nodes) > 0, "Should produce at least one node"
print(f"  Produced {len(nodes)} nodes  PASS")

# --- Test 2: semantic sections are represented ---
print("\n[Test 2] Section coverage...")
sections = {n.metadata.get("section") for n in nodes}
print(f"  Sections found: {sections}")
assert "header" in sections, "Missing 'header' section"
assert "experience" in sections, "Missing 'experience' section"
assert "education" in sections, "Missing 'education' section"
print("  All expected sections present  PASS")

# --- Test 3: no node is longer than chunk_size * 5 characters (sanity check) ---
# JSON content averages ~4-5 chars/token; use 5x to stay clear of false positives.
print("\n[Test 3] Node size sanity check...")
oversized = [n for n in nodes if len(n.get_content()) > config.CHUNK_SIZE * 5]
assert not oversized, f"{len(oversized)} nodes are suspiciously large"
print(f"  No oversized nodes  PASS")

# --- Test 4: vector database creation ---
print("\n[Test 4] Vector database creation (this calls the embedding API)...")
index = data_processing.create_vector_database(nodes)
assert index is not None
print("  Index created  PASS")

# --- Test 5: embedding verification ---
print("\n[Test 5] Embedding verification...")
result = data_processing.verify_embeddings(index)
assert result, "Some nodes are missing embeddings"
print("  All nodes embedded  PASS")

print("\n=== Phase 3: ALL TESTS PASSED ===")
