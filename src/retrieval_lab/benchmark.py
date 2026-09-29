"""Commands for data audit, public-model download, evaluation, and local prediction."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from .core import BM25, corpus_hash, load_corpus, load_queries, validate_splits


def audit_data(path):
    documents = load_corpus(path / "corpus.jsonl")
    dev = load_queries(path / "dev.jsonl", documents)
    test = load_queries(path / "test.jsonl", documents)
    validate_splits(dev, test)
    overlap = sorted({d for q in dev for d in q.relevant_ids} & {d for q in test for d in q.relevant_ids})
    if overlap:
        raise ValueError("gold document overlap between development and held-out test")
    return {"document_count": len(documents), "dev_count": len(dev), "test_count": len(test),
            "corpus_sha256": corpus_hash(documents), "gold_doc_overlap": overlap}


def download_models(path):
    from huggingface_hub import snapshot_download
    from .models import ENCODER_ID, ENCODER_REVISION, READER_ID, READER_REVISION
    for role, model, revision in [("encoder", ENCODER_ID, ENCODER_REVISION),
                                  ("reader", READER_ID, READER_REVISION)]:
        snapshot_download(model, revision=revision, local_dir=path / role,
                          allow_patterns=["*.json", "*.safetensors", "vocab.txt", "1_Pooling/*"], max_workers=2)


def machine_info():
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "cpu_count": os.cpu_count(), "device": "cpu", "torch_threads": 2, "gpu_used": False}
    if sys.platform == "darwin":
        for field, name in [("cpu_model", "machdep.cpu.brand_string"), ("memory_bytes", "hw.memsize")]:
            info[field] = subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    return info


def run(data, models, output, evidence):
    import pandas as pd
    from .core import calibrate_threshold, score_predictions
    from .models import Encoder, Reader, QABundle, directory_files, sha256_file
    from .tracking import log_and_reload

    if output.exists():
        raise ValueError("output directory already exists; choose a new directory to preserve previous runs")
    audit = audit_data(data)
    documents = load_corpus(data / "corpus.jsonl")
    dev = load_queries(data / "dev.jsonl", documents)
    test = load_queries(data / "test.jsonl", documents)
    started = time.perf_counter()
    encoder, reader = Encoder(models / "encoder"), Reader(models / "reader")
    bundle = QABundle(documents, encoder, reader, reader_k=3)
    build_seconds = time.perf_counter() - started
    report = {"schema_version": 1, "run_utc": datetime.now(timezone.utc).isoformat(),
              "data_audit": audit,
              "hardware": machine_info(), "build_seconds_excluding_download": build_seconds,
              "latency_protocol": {"warmup_queries_per_retriever": 1, "measurement": "one sequential pass",
                                   "queries_per_split": 12, "reader_k": 3, "retrieval_k": 5,
                                   "includes_query_encoding_and_reader": True,
                                   "excludes_download_load_index_warmup": True,
                                   "percentile_method": "linear interpolation",
                                   "limits": "12 samples per split; descriptive CPU measurements, no throughput claim"},
              "data_file_sha256": {p.name: sha256_file(p) for p in sorted(data.glob("*.jsonl"))},
              "models": {"encoder": {"files": directory_files(models / "encoder"),
                                     "fingerprint": encoder.fingerprint},
                         "reader": {"files": directory_files(models / "reader"),
                                    "fingerprint": reader.fingerprint}},
              "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
              "experiments": {}, "limitations": ["Original small synthetic English dataset, not production evidence",
                  "Dependency and model snapshots are recorded in this report",
                  "Extractive SQuAD 1 reader scores are not calibrated probabilities of answerability",
                  "No test-set tuning, confidence intervals, significance claim, GPU, or public service"]}
    metrics_for_tracking = {}
    for name, retriever in [("bm25", BM25(documents)), ("dense", bundle.index)]:
        print(f"Evaluating {name} on CPU", file=sys.stderr, flush=True)
        bundle.raw_predict(dev[0].question, retriever)  # excluded warmup
        def collect(queries):
            return [{"query_id": query.query_id, "question": query.question,
                     "answers": list(query.answers), "relevant_ids": list(query.relevant_ids),
                     **bundle.raw_predict(query.question, retriever)} for query in queries]
        dev_rows = collect(dev)
        calibration = calibrate_threshold(dev_rows, split="dev")
        threshold = calibration["threshold"]  # frozen before any held-out predictions
        test_rows = collect(test)
        experiment = {"calibration": calibration}
        for split, rows in [("dev", dev_rows), ("test", test_rows)]:
            metrics, scored = score_predictions(rows, threshold, reader_k=3)
            experiment[split] = {"metrics": metrics, "predictions": scored}
            metrics_for_tracking.update({f"{name}.{split}.{key}": value for key, value in metrics.items()})
        report["experiments"][name] = experiment
        if name == "dense":
            bundle.threshold = threshold
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    questions = pd.DataFrame({"question": [q.question for q in test]})
    report["mlflow"] = log_and_reload(bundle, output, metrics_for_tracking, questions,
                                      artifact_files=[evidence, data / "provenance.json"])
    report["bundle_manifest"] = json.loads((output / "bundle" / "manifest.json").read_text())
    evidence.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "benchmark.json").write_text(evidence.read_text())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--data", type=Path, default=Path("data"))
    download = subparsers.add_parser("download")
    download.add_argument("--models", type=Path, default=Path("artifacts/models"))
    evaluate = subparsers.add_parser("run")
    evaluate.add_argument("--data", type=Path, default=Path("data"))
    evaluate.add_argument("--models", type=Path, default=Path("artifacts/models"))
    evaluate.add_argument("--output", type=Path, default=Path("artifacts/run"))
    evaluate.add_argument("--evidence", type=Path, default=Path("evidence/benchmark.json"))
    predict = subparsers.add_parser("predict")
    predict.add_argument("--bundle", type=Path, default=Path("artifacts/run/bundle"))
    predict.add_argument("question")
    args = parser.parse_args()
    try:
        if args.command == "audit":
            print(json.dumps(audit_data(args.data), indent=2))
        elif args.command == "download":
            download_models(args.models)
        elif args.command == "predict":
            from .models import QABundle
            print(json.dumps(QABundle.load(args.bundle).predict(args.question), indent=2))
        elif args.command == "run":
            report = run(args.data, args.models, args.output, args.evidence)
            print(json.dumps({name: value["test"]["metrics"]
                              for name, value in report["experiments"].items()}, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
