# Local runbook

Run commands from the project root. `scripts/bootstrap.sh` creates `.bootstrap`, `.runtime`, `.cache`, and `.venv` locally. It requires an existing Python capable of creating a virtual environment and fetches uv 0.9.7 and Python 3.11.14. Dependency and model downloads use public package/model sources; no credentials are required.

## Standard cycle

```bash
./scripts/bootstrap.sh
.venv/bin/retrieval-eval audit
.venv/bin/retrieval-eval download
TOKENIZERS_PARALLELISM=false .venv/bin/retrieval-eval run \
  --output artifacts/run-02 --evidence evidence/run-02.json
```

The audit validates the complete dataset before model loading. A run saves raw and scored predictions, separate development/test metrics, calibration curves, exact corpus/model hashes, and a portable dense QA bundle. The output directory must be new.

The checked-in local evidence used `artifacts/final-run`. Paths in its MLflow record identify that local execution; after cloning, produce your own run instead of assuming those local artifacts are present.

## Inspect MLflow

```bash
.venv/bin/mlflow ui --backend-store-uri ./artifacts/final-run/mlruns \
  --host 127.0.0.1 --port 4330
```

Open `http://127.0.0.1:4330`. The experiment contains both retrievers' development/test metrics, model fingerprints, and the reload parity metric. The packaged model contains the dense pipeline; BM25 is the comparison baseline. Stop the UI with Ctrl-C. The experiment logs to local files and requires no server for evaluation.

## Offline prediction

After a complete run, use its bundle with the network disabled:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/retrieval-eval predict --bundle artifacts/final-run/bundle \
  'How long are visitor logs preserved in the Iris archive?'
```

Blank answers represent abstention. Accepted answers include a document ID and character offsets in that document's text. Changing a corpus, encoder file, reader file, embedding file, or document ordering requires rebuilding the bundle. Do not edit hashes to silence a mismatch.

## Haystack 1.x experiment

The following installs the independently locked environment tested on macOS ARM64:

```bash
UV_PYTHON_INSTALL_DIR="$PWD/.runtime" .bootstrap/bin/uv venv .venv-haystack --python 3.11
.bootstrap/bin/uv pip sync --python .venv-haystack/bin/python \
  experiments/haystack1/requirements.lock
HAYSTACK_TELEMETRY_ENABLED=false TOKENIZERS_PARALLELISM=false \
  .venv-haystack/bin/python -m unittest discover \
  -s experiments/haystack1 -p 'test_*.py'
```

This executes real BM25 retrieval, Sentence Transformers embeddings through `EmbeddingRetriever`, and `TransformersReader` over both splits, saving `evidence/haystack1.json`. It requires models from the main download step. Telemetry is disabled and inference uses local model files. The legacy dependency set is isolated from the main environment.

## Troubleshooting and verification

| Symptom | Action |
|---|---|
| Python 3.13 fails older package installation | Run the bootstrap; execution is pinned to Python 3.11. |
| Model files missing | Run `retrieval-eval download` before integration tests or the benchmark. |
| An output directory already exists | Select a fresh `--output`; preserve the previous evidence. |
| Bundle mismatch | Rebuild from the intended corpus and pinned models; inspect the manifest. |
| Sparse retrieval returns no documents | The pipeline returns an empty answer with zero confidence; this is a tested path. |
| Too much RAM/disk consumed | Keep only required generated runs after reviewing them; each MLflow run stores model files. |
| Haystack resolver conflict | Use its separate environment and exact requirements; do not merge with the main lockfile. |

Run `RUN_MODEL_TESTS=1 TOKENIZERS_PARALLELISM=false .venv/bin/pytest -q` for the full main suite. The normal CI job skips the real-model marker and does not download weights; manually dispatch the real-model workflow for the bounded integration run. Hosted CI and Linux execution have not been observed locally.
