# Sources and model attribution

This project implements its own evaluation and persistence code; pretrained weights come from the model publishers.

| Source | Used for |
|---|---|
| [Dense Passage Retrieval paper](https://arxiv.org/abs/2004.04906) | Separation of retrieval evaluation from downstream extractive QA. The project uses MiniLM, not DPR checkpoints. |
| [Sentence-BERT paper](https://arxiv.org/abs/1908.10084) | Sentence embedding retrieval context. |
| [Sentence Transformers v2.0.0 model documentation](https://github.com/huggingface/sentence-transformers/blob/v2.0.0/docs/pretrained_models.md) | Archived 2021 documentation explicitly names `paraphrase-MiniLM-L6-v2`. |
| [MiniLM model card](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2) | Mean pooling, dimensionality, model limits, and Apache-2.0 model license. |
| [Transformers v3.0.2 pipeline source](https://github.com/huggingface/transformers/blob/v3.0.2/src/transformers/pipelines.py) | Archived 2020 QA pipeline and DistilBERT model support. |
| [DistilBERT SQuAD model card](https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad) | Extractive reader training task, checkpoint identity, and Apache-2.0 license. |
| [Haystack v1.0.0 release](https://github.com/deepset-ai/haystack/releases/tag/v1.0.0) | The 2021 retriever/reader pipeline API lineage. |
| [Haystack 1.26.3 package metadata](https://pypi.org/project/farm-haystack/1.26.3/) | Compatible execution package and dependency declaration. |
| [Haystack 1.x end-of-life notice](https://github.com/deepset-ai/haystack/discussions/8935) | Maintenance status of the isolated legacy integration. |
| [MLflow 2.17.2 pyfunc API](https://mlflow.org/docs/2.17.2/python_api/mlflow.pyfunc.html) | Custom Python model packaging and loading. |

Execution dependencies are listed in the README and lockfiles. Archived source anchors establish which techniques and model names were available in 2020–2021; they do not identify the byte hashes of today's downloaded model snapshots. The current pinned snapshots include safetensors packaging and updated metadata. Exact revisions and all file hashes are recorded in the benchmark and bundle manifests.

| Model | Pinned revision |
|---|---|
| `sentence-transformers/paraphrase-MiniLM-L6-v2` | `c9a2bfebc254878aee8c3aca9e6844d5bbb102d1` |
| `distilbert/distilbert-base-cased-distilled-squad` | `564e9b582944a57a3e586bbb98fd6f0a4118db7f` |
