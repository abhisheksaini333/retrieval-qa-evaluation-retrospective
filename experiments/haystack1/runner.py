"""Independent Haystack 1.x execution; never import this into the main environment."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def evaluate(root, output):
    root = root.resolve()
    sys.path.insert(0, str(root / "src"))
    import torch
    from haystack.document_stores import InMemoryDocumentStore
    from haystack.nodes import BM25Retriever, EmbeddingRetriever, TransformersReader
    from retrieval_lab.core import load_corpus, load_queries, validate_splits, corpus_hash
    from retrieval_lab.core import calibrate_threshold, score_predictions
    from retrieval_lab.models import directory_fingerprint

    torch.set_num_threads(2)
    docs = load_corpus(root / "data/corpus.jsonl")
    dev = load_queries(root / "data/dev.jsonl", docs)
    test = load_queries(root / "data/test.jsonl", docs)
    validate_splits(dev, test)
    store = InMemoryDocumentStore(embedding_dim=384, similarity="cosine", use_gpu=False,
                                  use_bm25=True, duplicate_documents="fail", progress_bar=False)
    # Both retrievers consume the same title + text. The legacy reader also receives
    # title + text, unlike the main benchmark's text-only reader context.
    store.write_documents([{"id": d.doc_id, "content": d.title + " " + d.text} for d in docs])
    encoder = root / "artifacts/models/encoder"
    reader_path = root / "artifacts/models/reader"
    dense = EmbeddingRetriever(embedding_model=str(encoder), document_store=store,
                               model_format="sentence_transformers", use_gpu=False,
                               max_seq_len=128, progress_bar=False, top_k=5, scale_score=False)
    store.update_embeddings(dense)
    sparse = BM25Retriever(document_store=store, top_k=5, scale_score=False)
    reader = TransformersReader(model_name_or_path=str(reader_path), use_gpu=False,
                                max_seq_len=384, top_k_per_candidate=1, return_no_answers=False)
    report = {"schema_version": 1, "status": "executed", "run_utc": datetime.now(timezone.utc).isoformat(),
              "packages": {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()},
              "hardware": {"platform": platform.platform(), "device": "cpu", "torch_threads": 2},
              "corpus_sha256": corpus_hash(docs), "model_fingerprints": {
                  "encoder": directory_fingerprint(encoder), "reader": directory_fingerprint(reader_path)},
              "protocol": {"calibration_split": "dev", "reader_k": 3, "retrieval_k": 5,
                           "warmup_queries": 1, "measurement": "single sequential pass per split",
                           "reader_context": "title + text", "main_reader_context": "text only",
                           "comparability": "Independent adapter validation; not a numerical parity claim"},
              "experiments": {}}
    def predict(query, retriever):
        start = time.perf_counter()
        ranked = retriever.retrieve(query.question, top_k=5)
        retrieved = time.perf_counter()
        answer = reader.predict(query.question, documents=ranked[:3], top_k=1)["answers"]
        finished = time.perf_counter()
        best = answer[0] if answer else None
        return {"query_id": query.query_id, "question": query.question, "answers": list(query.answers),
                "relevant_ids": list(query.relevant_ids), "ranked_ids": [d.id for d in ranked],
                "ranked_scores": [float(d.score) for d in ranked],
                "raw_answer": best.answer if best else "", "confidence": float(best.score) if best else 0.0,
                "document_id": best.document_ids[0] if best and best.document_ids else None,
                "retrieval_ms": (retrieved - start) * 1000, "reader_ms": (finished - retrieved) * 1000,
                "total_ms": (finished - start) * 1000}
    for name, retriever in [("bm25", sparse), ("dense", dense)]:
        predict(dev[0], retriever)
        dev_rows = [predict(query, retriever) for query in dev]
        calibration = calibrate_threshold(dev_rows, split="dev")
        test_rows = [predict(query, retriever) for query in test]
        result = {"calibration": calibration}
        for split, rows in [("dev", dev_rows), ("test", test_rows)]:
            metrics, predictions = score_predictions(rows, calibration["threshold"], reader_k=3)
            result[split] = {"metrics": metrics, "predictions": predictions}
        report["experiments"][name] = result
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("evidence/haystack1.json"))
    args = parser.parse_args()
    result = evaluate(args.root, args.output)
    print(json.dumps({name: data["test"]["metrics"] for name, data in result["experiments"].items()}, indent=2))
