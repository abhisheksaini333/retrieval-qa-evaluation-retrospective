"""Safe experiment execution and reproducible run protocols."""
from contextlib import contextmanager
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
    evidence.parent.mkdir(parents=True, exist_ok=True)
    lock = evidence.with_name(evidence.name + ".lock")
    with lock.open("x"):
        try:
            output.mkdir(parents=True, exist_ok=False)
            yield output
        finally:
            lock.unlink()
