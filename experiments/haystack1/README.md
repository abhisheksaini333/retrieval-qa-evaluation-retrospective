# Haystack 1.x adapter

`runner.py` evaluates `BM25Retriever` and real `EmbeddingRetriever` outputs through `TransformersReader`, calibrates each threshold on development questions, and writes held-out metrics and individual predictions.

Use the separate environment in the [runbook](../../docs/runbook.md). `requirements.in` declares the compatible versions; `requirements.lock` records the tested macOS ARM64 resolution. The [saved report](../../evidence/haystack1.json) and [execution log](../../evidence/haystack1-run.log) are from a successful local CPU run.

Reader input is title plus text in this adapter. It is text only in the main benchmark, so this is a separate integration experiment. No API calls or hosted vector database are needed.
