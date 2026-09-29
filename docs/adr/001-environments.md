# ADR 001: Isolate the main pipeline and Haystack 1.x

Status: accepted.

The main pipeline directly calls Transformers 4.44.2 and uses MLflow 2.17.2. The Haystack adapter uses farm-haystack 1.26.3, whose published dependency metadata requires Transformers 4.39.3. Keeping these in separate virtual environments avoids incompatible resolver requirements and makes each API surface inspectable.

Haystack 1.26.4 changes its Transformers requirement to `>=4.46,<5.0`; the original 1.26.4/4.39.3 combination failed dependency resolution. The selected 1.26.3/4.39.3 combination resolved, imported, and completed the integration benchmark. Exact transitive versions and distribution hashes are recorded in `experiments/haystack1/requirements.lock`; the main dependency graph is in `uv.lock`.

Both environments run Python 3.11 and Torch 2.5.1 on CPU. A project-local bootstrap installs the runtime without changing system Python or executable links. Linux resolves the main environment's Torch wheel from the explicit official CPU index. The Haystack hash lock was resolved and tested on macOS ARM64; Linux users should resolve the documented input requirements in their isolated environment before claiming portability.

The encoder uses masked mean pooling and normalized cosine similarity, with the model card's 128-token limit. The reader uses the SQuAD DistilBERT checkpoint. Full saved model files accompany the index so MLflow reload is independent of network availability. SHA-256 checks detect accidental mismatches; they are not artifact signatures or a substitute for trusting the source of executable MLflow model packages.
