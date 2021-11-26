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
    from .core import validate_evidence_isolation
    validate_evidence_isolation(dev, test, documents)
    overlap = sorted({d for q in dev for d in q.relevant_ids} & {d for q in test for d in q.relevant_ids})
    if overlap:
        raise ValueError("gold document overlap between development and held-out test")
    from .datasets import validate_provenance, dataset_manifest
    validate_provenance(json.loads((path / "provenance.json").read_text()),
                        {"documents": len(documents), "development_queries": len(dev),
                         "held_out_queries": len(test)})
    return {"document_count": len(documents), "dev_count": len(dev), "test_count": len(test),
            "corpus_sha256": corpus_hash(documents), "gold_doc_overlap": overlap,
            "dataset_manifest": dataset_manifest(path)}


def download_models(path):
    from huggingface_hub import snapshot_download
    from .models import ENCODER_ID, ENCODER_REVISION, READER_ID, READER_REVISION
    for role, model, revision in [("encoder", ENCODER_ID, ENCODER_REVISION),
                                  ("reader", READER_ID, READER_REVISION)]:
        snapshot_download(model, revision=revision, local_dir=path / role,
                          allow_patterns=["*.json", "*.safetensors", "vocab.txt", "1_Pooling/*"], max_workers=2)


def machine_info():
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "cpu_count": os.cpu_count(), "device": "cpu",
            "torch_threads": __import__("torch").get_num_threads(), "gpu_used": False}
    if sys.platform == "darwin":
        for field, name in [("cpu_model", "machdep.cpu.brand_string"), ("memory_bytes", "hw.memsize")]:
            info[field] = subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    return info


def run(data, models, output, evidence, config=None):
    from .execution import reserve_run, RunConfig, run_status
    config = config if config is not None else RunConfig()
    if not isinstance(config, RunConfig):
        raise ValueError("config must be RunConfig")
    with reserve_run(output, evidence, [data, models]), run_status(output):
        return _run(data, models, output, evidence, config)


def _run(data, models, output, evidence, config):
    import pandas as pd
    from .core import calibrate_threshold, score_predictions
    from .models import Encoder, Reader, QABundle, directory_files, sha256_file
    from .tracking import log_and_reload

    audit = audit_data(data)
    documents = load_corpus(data / "corpus.jsonl")
    dev = load_queries(data / "dev.jsonl", documents)
    test = load_queries(data / "test.jsonl", documents)
    from .execution import set_stage
    set_stage(output, "model_load")
    started = time.perf_counter()
    encoder, reader = Encoder(models / "encoder"), Reader(models / "reader")
    bundle = QABundle(documents, encoder, reader, reader_k=config.reader_k)
    build_seconds = time.perf_counter() - started
    report = {"schema_version": 1, "run_utc": datetime.now(timezone.utc).isoformat(),
              "data_audit": audit,
              "hardware": machine_info(), "build_seconds_excluding_download": build_seconds,
              "latency_protocol": config.protocol(len(dev), len(test)),
              "data_file_sha256": {p.name: sha256_file(p) for p in sorted(data.glob("*.jsonl"))},
              "models": {"encoder": {"files": directory_files(models / "encoder"),
                                     "fingerprint": encoder.fingerprint},
                         "reader": {"files": directory_files(models / "reader"),
                                    "fingerprint": reader.fingerprint}},
              "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
              "experiments": {}, "limitations": ["Original small synthetic English dataset, not production evidence",
                  "Dependency and model snapshots are recorded in this report",
                  "Extractive SQuAD 1 reader scores are not calibrated probabilities of answerability",
                  "No test-set tuning, significance claim, GPU, or public service"]}
    metrics_for_tracking = {}
    from .execution import collect_predictions
    from .retrieval import FusionRetriever
    from .metrics import bootstrap_mean
    engines = [("bm25", BM25(documents)), ("dense", bundle.index)]
    if config.include_fusion:
        engines.append(("fusion", FusionRetriever([engine for _, engine in engines])))
    for name, retriever in engines:
        set_stage(output, f"evaluate_{name}")
        print(f"Evaluating {name} on CPU", file=sys.stderr, flush=True)
        for index in range(config.warmups):
            bundle.raw_predict(dev[index % len(dev)].question, retriever, retrieval_k=config.retrieval_k)
        dev_rows = collect_predictions(bundle, dev, retriever, config)
        calibration = calibrate_threshold(dev_rows, split="dev")
        threshold = calibration["threshold"]  # frozen before any held-out predictions
        test_rows = collect_predictions(bundle, test, retriever, config)
        experiment = {"calibration": calibration}
        for split, rows in [("dev", dev_rows), ("test", test_rows)]:
            metrics, scored = score_predictions(rows, threshold, reader_k=config.reader_k,
                                               cutoffs=tuple(k for k in (1, 3, 5) if k <= config.retrieval_k))
            experiment[split] = {"metrics": metrics, "predictions": scored,
                                 "em_bootstrap": bootstrap_mean([row["answer_em"] for row in scored],
                                                                 repetitions=1000, seed=config.seed)}
            metrics_for_tracking.update({f"{name}.{split}.{key}": value for key, value in metrics.items() if value is not None})
        report["experiments"][name] = experiment
        if name == "dense":
            bundle.threshold = threshold
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    questions = pd.DataFrame({"question": [q.question for q in test]})
    set_stage(output, "tracking_and_reload")
    report["mlflow"] = log_and_reload(bundle, output, metrics_for_tracking, questions,
                                      artifact_files=[data / "provenance.json"])
    report["bundle_manifest"] = json.loads((output / "bundle" / "manifest.json").read_text())
    evidence.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "benchmark.json").write_text(evidence.read_text())
    from .tracking import log_final_benchmark
    log_final_benchmark(evidence, report["mlflow"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--data", type=Path, default=Path("data"))
    download = subparsers.add_parser("download")
    download.add_argument("--models", type=Path, default=Path("artifacts/models"))
    evaluate = subparsers.add_parser("run")
    for flag, default in [("reader-k", 3), ("retrieval-k", 5), ("warmups", 1), ("repeats", 1), ("seed", 0)]:
        evaluate.add_argument("--" + flag, type=int, default=default)
    evaluate.add_argument("--fusion", action="store_true")
    evaluate.add_argument("--data", type=Path, default=Path("data"))
    evaluate.add_argument("--models", type=Path, default=Path("artifacts/models"))
    evaluate.add_argument("--output", type=Path, default=Path("artifacts/run"))
    evaluate.add_argument("--evidence", type=Path, default=Path("evidence/benchmark.json"))
    for command in ("export", "import"):
        archive = subparsers.add_parser(command)
        archive.add_argument("--bundle", type=Path, required=True)
        archive.add_argument("--archive", type=Path, required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--bundle", type=Path, required=True)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--bundle", type=Path, default=Path("artifacts/run/bundle"))
    predict.add_argument("question")
    args = parser.parse_args()
    try:
        if args.command == "audit":
            print(json.dumps(audit_data(args.data), indent=2))
        elif args.command in {"export", "import"}:
            from .bundle_io import export_bundle, import_bundle
            result = (export_bundle(args.bundle, args.archive) if args.command == "export"
                      else import_bundle(args.archive, args.bundle))
            print(json.dumps(result, indent=2))
        elif args.command == "inspect":
            from .bundle_io import inspect_bundle
            print(json.dumps(inspect_bundle(args.bundle), indent=2))
        elif args.command == "download":
            download_models(args.models)
        elif args.command == "predict":
            from .models import QABundle
            print(json.dumps(QABundle.load(args.bundle).predict(args.question), indent=2))
        elif args.command == "run":
            from .execution import RunConfig
            config = RunConfig(args.reader_k, args.retrieval_k, args.warmups, args.repeats, args.seed, args.fusion)
            report = run(args.data, args.models, args.output, args.evidence, config)
            print(json.dumps({name: value["test"]["metrics"]
                              for name, value in report["experiments"].items()}, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
