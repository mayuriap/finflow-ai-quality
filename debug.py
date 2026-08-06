import sys
sys.path.insert(0, '.')

print("Step 1: starting", flush=True)

try:
    from src.llm_validator.output_validator import LLMOutputValidator
    print("Step 2: import ok", flush=True)
except Exception as e:
    print(f"Step 2 FAILED: {e}", flush=True)
    sys.exit(1)

try:
    v = LLMOutputValidator()
    print("Step 3: validator created", flush=True)
except Exception as e:
    print(f"Step 3 FAILED: {e}", flush=True)
    sys.exit(1)

try:
    response = v.client.ask("Say hello in one word")
    print(f"Step 4: LLM responded: {response}", flush=True)
except Exception as e:
    print(f"Step 4 FAILED: {e}", flush=True)
    sys.exit(1)

print("All steps passed!", flush=True)