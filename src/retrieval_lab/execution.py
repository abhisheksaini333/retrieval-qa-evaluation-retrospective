"""Safe experiment execution and reproducible run protocols."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import statistics
import json
from datetime import datetime, timezone
from pathlib import Path
from .bundle_io import validate_output_location


@contextmanager
def reserve_run(output, evidence, sources):
    output = validate_output_location(output, sources)
    evidence = validate_output_location(evidence, sources)
    if output.exists() or evidence.exists():
        raise FileExistsError("run output or evidence already exists")
    if evidence == output or evidence in output.parents:
        raise ValueError("evidence path cannot contain the run directory")
    nested = output in evidence.parents
    if nested:
        output.mkdir(parents=True, exist_ok=False)
    evidence.parent.mkdir(parents=True, exist_ok=True)
    lock = evidence.with_name(evidence.name + ".lock")
    with lock.open("x"):
        try:
            if not nested:
                output.mkdir(parents=True, exist_ok=False)
            yield output
        finally:
            lock.unlink()


@dataclass(frozen=True)
class RunConfig:
    reader_k: int = 3
    retrieval_k: int = 5
    warmups: int = 1
    repeats: int = 1
    seed: int = 0
    include_fusion: bool = False

    def __post_init__(self):
        if (type(self.reader_k) is not int or type(self.retrieval_k) is not int
                or not 1 <= self.reader_k <= self.retrieval_k <= 100
                or type(self.warmups) is not int or not 0 <= self.warmups <= 100
                or type(self.repeats) is not int or not 1 <= self.repeats <= 100
                or type(self.seed) is not int or type(self.include_fusion) is not bool):
            raise ValueError("invalid run configuration")

    def protocol(self, dev_count, test_count):
        return {**asdict(self), "queries_per_split": {"dev": dev_count, "test": test_count},
                "measurement": "per-query median across sequential repetitions",
                "warmup_queries_per_retriever": self.warmups,
                "includes_query_encoding_and_reader": True, "excludes_download_load_index_warmup": True,
                "percentile_method": "linear interpolation", "device": "cpu"}


def collect_predictions(bundle, queries, retriever, config):
    rows = []
    for query in queries:
        samples = [bundle.raw_predict(query.question, retriever, retrieval_k=config.retrieval_k)
                   for _ in range(config.repeats)]
        timing_keys = {"retrieval_ms", "reader_ms", "total_ms"}
        stable = [{k: v for k, v in sample.items() if k not in timing_keys} for sample in samples]
        if any(value != stable[0] for value in stable[1:]):
            raise ValueError("predictions changed between measurement repetitions")
        rows.append({"query_id": query.query_id, "question": query.question,
                     "answers": list(query.answers), "relevant_ids": list(query.relevant_ids), **samples[0],
                     **{key: statistics.median(s[key] for s in samples) for key in timing_keys},
                     "timing_samples": [{key: sample[key] for key in sorted(timing_keys)} for sample in samples]})
    return rows


def set_stage(output, stage, state="running", **details):
    output = Path(output)
    record = {"state": state, "stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(), **details}
    temporary = output / "status.json.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(output / "status.json")


@contextmanager
def run_status(output):
    set_stage(output, "audit")
    try:
        yield
    except BaseException as error:
        previous = json.loads((Path(output) / "status.json").read_text())
        set_stage(output, previous["stage"], "failed", error_type=type(error).__name__, error=str(error)[:1000])
        raise
    else:
        set_stage(output, "complete", "complete")
