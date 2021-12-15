"""Compare benchmark evidence while checking paired-evaluation assumptions."""
from .metrics import paired_comparison


def compare_benchmarks(left, right, *, metric="answer_em", seed=0, repetitions=1000):
    for report in (left, right):
        if not isinstance(report, dict) or report.get("schema_version") != 1:
            raise ValueError("unsupported benchmark schema")
        if not all(report.get(key) for key in ("data_file_sha256", "latency_protocol", "models", "experiments")):
            raise ValueError("benchmark is missing comparison evidence")
    for key in ("data_file_sha256", "latency_protocol"):
        if left[key] != right[key]:
            raise ValueError(f"benchmarks have incompatible {key}")
    if left.get("execution") != right.get("execution"):
        raise ValueError("benchmarks have incompatible inference settings")
    for role in ("encoder", "reader"):
        if left["models"][role]["fingerprint"] != right["models"][role]["fingerprint"]:
            raise ValueError(f"benchmarks have incompatible {role} models")
    shared = sorted(left["experiments"].keys() & right["experiments"].keys())
    if not shared:
        raise ValueError("benchmarks have no shared retrieval experiment")
    experiments = {name: paired_comparison(left["experiments"][name]["test"]["predictions"],
                                           right["experiments"][name]["test"]["predictions"],
                                           metric=metric, seed=seed, repetitions=repetitions) for name in shared}
    return {"direction": "left minus right", "metric": metric, "experiments": experiments,
            "latency_comparable": bool(left.get("hardware")) and left.get("hardware") == right.get("hardware"),
            "limits": "Paired bootstrap describes this dataset; it is not evidence of production performance."}
