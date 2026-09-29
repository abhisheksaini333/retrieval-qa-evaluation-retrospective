# Retrieval QA evaluation implementation plan

**Goal:** Compare BM25 and dense retrieval, inspect reader failures, calibrate abstention, and prove artifact reload parity.

**Architecture:** Validate an original synthetic corpus and disjoint development/held-out queries; compare BM25 against a real MiniLM encoder, apply the same extractive DistilBERT reader, calibrate rejection on development only, freeze thresholds, evaluate held-out questions. Save content-checked bundles and prove MLflow reload parity. Run the separately pinned Haystack 1.x experiment when its isolated dependencies are available.

**Tech stack:** Python 3.11, NumPy, Transformers 4.44.2, Torch 2.5.1, MLflow 2.17.2; Haystack experiment uses farm-haystack 1.26.3 and Transformers 4.39.3.

Design approved through the parent task. Work occurs in this directory without Git operations; no extra worktree or approval is needed.

1. Write tests in `tests/test_core.py` for malformed data, duplicate IDs, empty retrieval, metric denominators, development-only calibration, stable BM25 ordering, and index mismatch. Run RED, implement `src/retrieval_lab/core.py`, run GREEN.
2. Write `tests/test_models.py` for real dense inference, bundle hash rejection, reload prediction parity, and MLflow round trip. Require explicit integration mode and actual model downloads. Implement `models.py`, `tracking.py`, and `benchmark.py`; run full CPU experiment.
3. Add a separately pinned `experiments/haystack1` adapter with executable smoke validation. Capture actual installed/runtime status without claiming unexecuted integrations.
4. Save benchmark JSON, dependency locks, machine/runtime limits, provenance, runbook, ADR, MIT license, and CI. Verify tests, benchmark evidence, and link integrity before handoff.
