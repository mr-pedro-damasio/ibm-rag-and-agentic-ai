"""
Phase 5 test — full end-to-end pipeline test.
Verifies:
  1. main.py --mock runs to the chatbot prompt and exits cleanly.
  2. --model flag overrides the LLM and the pipeline still works.
  3. The stray xml import is gone.
  4. llm_interface.py is gone.

Run with:
    python test_phase5.py
"""
import sys
import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# --- Test 1: stray import is gone from main.py ---
print("[Test 1] No stray xml import in main.py...")
with open("main.py") as f:
    source = f.read()
assert "xml.parsers.expat" not in source, \
    "Stray 'from xml.parsers.expat import model' still in main.py"
print("  PASS")

# --- Test 2: llm_interface.py is gone ---
print("\n[Test 2] llm_interface.py is deleted...")
assert not os.path.exists("llm_interface.py"), \
    "llm_interface.py still exists"
print("  PASS")

# --- Test 3: main.py --mock full pipeline ---
print("\n[Test 3] Full pipeline with --mock (sends 'exit' to chatbot)...")
result = subprocess.run(
    [sys.executable, "main.py", "--mock", "--log-level", "WARNING"],
    input="exit\n",
    capture_output=True, text=True, timeout=120
)
print(f"  stdout preview: {result.stdout[:200].strip()}")
if result.returncode != 0:
    print(f"  stderr: {result.stderr[-500:]}")
assert result.returncode == 0, f"main.py exited with code {result.returncode}"
assert "interesting facts" in result.stdout.lower(), \
    "Output does not contain 'interesting facts'"
print("  PASS")

# --- Test 4: --model flag override ---
print("\n[Test 4] --model flag overrides the LLM...")
result = subprocess.run(
    [sys.executable, "main.py", "--mock",
     "--model", "openai/gpt-4o-mini",
     "--log-level", "INFO"],
    input="exit\n",
    capture_output=True, text=True, timeout=120
)
assert result.returncode == 0, f"--model run failed: {result.stderr[-500:]}"
# The INFO log should show the overridden model name
assert "gpt-4o-mini" in result.stderr, \
    "Expected to see model name in INFO logs"
print("  PASS")

print("\n=== Phase 5: ALL TESTS PASSED ===")
print("\nAll phases complete. The refactor is done.")
