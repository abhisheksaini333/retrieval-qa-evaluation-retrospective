"""Run after building a wheel: python scripts/verify_wheel.py artifacts/wheels/*.whl."""
import json
import sys
from retrieval_lab.packaging import verify_wheel

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_wheel.py WHEEL")
    print(json.dumps(verify_wheel(sys.argv[1]), indent=2))
