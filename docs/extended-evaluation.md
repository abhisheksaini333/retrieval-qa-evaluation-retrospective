# Evaluation and portable bundles

Run the fixed dataset with optional reciprocal-rank fusion, repeated latency measurements, and a seeded paired bootstrap:

```sh
retrieval-eval run --output artifacts/experiment-a --evidence evidence/experiment-a.json \
  --reader-k 3 --retrieval-k 5 --warmups 2 --repeats 3 --seed 42 --fusion
retrieval-eval inspect --bundle artifacts/experiment-a/bundle
retrieval-eval export --bundle artifacts/experiment-a/bundle --archive artifacts/qa-bundle.zip
retrieval-eval import --archive artifacts/qa-bundle.zip --bundle artifacts/imported
retrieval-eval compare evidence/experiment-a.json evidence/experiment-b.json
```

Choose unused output and evidence paths. A run reserves both paths, records its current stage in `status.json`, and preserves failure diagnostics. Latency is the per-query median across repetitions; the raw timing samples remain in the predictions. The protocol records actual split sizes. Calibration uses development queries only. Held-out answer EM and F1, retrieval coverage, and failure classes retain explicit denominators. Citation diagnostics are available through the Python metrics module. Undefined rates use JSON null. Small synthetic datasets support regression checks, not production performance claims.

`compare` checks data hashes, inference protocol, model fingerprints, and query coverage before a paired bootstrap of left-minus-right answer scores. Different hardware disables latency comparability. Bootstrap intervals describe sampling variability in this fixed dataset; they do not establish external validity.

For JSONL predictions, give each request a stable ID:

```json
{"query_id":"example-1","question":"How long are receipts retained?"}
```

```sh
retrieval-eval predict-batch --bundle artifacts/imported \
  --input requests.jsonl --output predictions.jsonl
```

Batch output preserves order and IDs. Validation and ordinary inference failures leave no partial output. `--continue-on-error` writes structured per-row inference errors. Existing files are preserved.

Bundles record actual model content identities and encoder pooling/token/batch configuration. Inspection validates manifest fields, corpus order, model inventories, hashes, and bounded NPY headers without importing Torch or Transformers. Loading additionally validates vector norms and dimensions. Schema 1 bundles retain their original fixed inference settings; new bundles use schema 2. Exports contain weights, so size and storage costs are similar to the model cache. Imports reject path traversal, symlinks, duplicate members, and oversized payloads. Checksums detect corruption; they do not authenticate a publisher.

MLflow output includes an explicit nullable schema, the exact dependency lock, exact reload predictions, and an isolated offline Python subprocess proof. The subprocess imports the artifact's packaged code. Tracking restores the caller's URI and rejects invocation inside an already-active run. When calling `log_and_reload` from an installed wheel, supply `dependency_lock=Path(...)` to the lock used for that environment.

The Python modules also expose Unicode retrieval tokenization, BM25 explanations and title weighting, cached/batched retrieval, MMR selection, chunk provenance, oracle-context reader diagnostics, risk-constrained calibration, reliability bins, and grouped split helpers. These are opt-in experiments; the default benchmark still uses BM25 and the real dense encoder. `RunConfig` controls benchmark depth, repetition, warmup, seed, and fusion. Encoder pooling experiments use `EncoderConfig` and are persisted with the bundle.

CI runs the lightweight suite without model downloads, builds a wheel, and exercises its CLI and BM25 implementation in isolated Python outside the checkout. The manually triggered real-model job additionally downloads the pinned models and checks inference, bundle reload, and MLflow parity.

`metrics.reciprocal_rank(ranked, relevant, k=5)` returns the first relevant reciprocal rank, zero for no hit, and `None` for an unanswerable query. Duplicate ranking/relevance IDs are rejected.

`metrics.hit_rate([(ranked, relevant), ...], k=5)` reports the share of answerable examples with any retrieved evidence and the evaluated sample count. Unanswerable examples are excluded explicitly.

`metrics.weighted_mean(values, weights)` supports explicit query weighting. Weights must be finite and nonnegative with positive total; values and weights must have equal cardinality. Report weighting choices alongside results.

Calibration dataset and development fingerprints must be lowercase 64-character SHA-256 hex digests. Supply `dataset_manifest(path)["dataset_sha256"]`; placeholder names such as `dataset-a` are rejected.

`datasets.grouped_split` requires explicit trimmed string group keys. Convert domain identifiers deliberately before calling; integers and strings are not silently merged by coercion. Existing string grouping remains deterministic for a fixed seed.

Batch prediction summaries contain `success_count` and `error_count` in addition to request count. `continue_on_error` must be a boolean; partial output is visible in the summary.
