"""Local MLflow tracking and actual pyfunc serialization/reload parity."""
from contextlib import contextmanager
from pathlib import Path
import importlib.metadata

import mlflow
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema
import pandas as pd

from .models import QABundle


class RetrievalPythonModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.bundle = QABundle.load(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        if (not isinstance(model_input, pd.DataFrame) or not model_input.columns.is_unique
                or set(model_input.columns) != {"question"}):
            raise ValueError("prediction input requires a dataframe with a question column")
        if model_input.empty:
            raise ValueError("prediction input cannot be empty")
        from .core import validate_request
        for question in model_input["question"]:
            validate_request(question, 1)
        return prediction_frame([self.bundle.predict(q) for q in model_input["question"]], index=model_input.index)


def log_and_reload(bundle, output, metrics, questions, artifact_files=(), dependency_lock=None):
    if mlflow.active_run() is not None:
        raise ValueError("cannot start an evaluation inside an active MLflow run")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dependency_lock = Path(dependency_lock) if dependency_lock else Path(__file__).resolve().parents[2] / "uv.lock"
    lock_metadata = package_dependency_lock(dependency_lock, output)
    bundle_path = output / "bundle"
    bundle.save(bundle_path)
    expected = prediction_frame([bundle.predict(q) for q in questions["question"]], index=questions.index)
    with local_tracking((output / "mlruns").as_uri()) as experiment_id:
        with mlflow.start_run(run_name="dense-qa-cpu", experiment_id=experiment_id) as run:
            mlflow.log_params({"reader_k": bundle.reader_k, "threshold": bundle.threshold,
                               "encoder_sha256": bundle.encoder.fingerprint,
                               "reader_sha256": bundle.reader.fingerprint, "device": "cpu",
                               "transformers_version": importlib.metadata.version("transformers"),
                               "torch_version": importlib.metadata.version("torch"),
                               "mlflow_version": importlib.metadata.version("mlflow")})
            mlflow.log_metrics(metrics)
            for file in artifact_files:
                mlflow.log_artifact(str(file), artifact_path="benchmark")
            model = mlflow.pyfunc.log_model(
                artifact_path="qa_model", python_model=RetrievalPythonModel(),
                artifacts={"bundle": str(bundle_path), "dependency_lock": str(output / "uv.lock")},
                code_paths=[str(Path(__file__).resolve().parent)],
                signature=prediction_signature(), input_example=questions,
                pip_requirements=["numpy==1.26.4", "pandas==2.2.3", "torch==2.5.1",
                                  "transformers==4.44.2", "huggingface-hub==0.26.2", "mlflow==2.17.2"],
            )
            loaded = mlflow.pyfunc.load_model(model.model_uri)
            observed = loaded.predict(questions)
            pd.testing.assert_frame_equal(expected, observed, check_exact=True)
            fresh = verify_fresh_process(model.model_uri, mlflow.get_tracking_uri(), questions, json_records(expected), output)
            mlflow.log_metric("reload_prediction_parity", 1.0)
            return {"tracking_uri": mlflow.get_tracking_uri(), "run_id": run.info.run_id,
                    "model_uri": model.model_uri, "prediction_parity": True, "dependency_lock": lock_metadata,
                    "fresh_process": fresh,
                    "comparison": "exact dataframe equality: answer, score, abstention, IDs, offsets",
                    "loaded_prediction": json_records(observed)}


def json_records(frame):
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def prediction_frame(rows, index=None):
    frame = pd.DataFrame(rows, index=index)
    for field in ("start", "end"):
        if field in frame:
            frame[field] = frame[field].astype("float64")
    return frame


def prediction_signature():
    return ModelSignature(inputs=Schema([ColSpec("string", "question")]), outputs=Schema([
        ColSpec("string", "answer"), ColSpec("double", "confidence"),
        ColSpec("boolean", "abstained"), ColSpec("string", "document_id", required=False),
        ColSpec("double", "start", required=False), ColSpec("double", "end", required=False),
        ColSpec("string", "retrieved_ids")]))


@contextmanager
def local_tracking(uri):
    if mlflow.active_run() is not None:
        raise ValueError("cannot replace tracking context during an active MLflow run")
    previous = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=uri)
        experiment = client.get_experiment_by_name("retrieval-qa-evaluation")
        experiment_id = experiment.experiment_id if experiment else client.create_experiment("retrieval-qa-evaluation")
        yield experiment_id
    finally:
        mlflow.set_tracking_uri(previous)


def log_final_benchmark(path, tracking):
    import json
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if not {"mlflow", "bundle_manifest"} <= report.keys() or report["mlflow"] != tracking:
        raise ValueError("benchmark must include final reload evidence")
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking["tracking_uri"])
    client.log_artifact(tracking["run_id"], str(path), artifact_path="benchmark")


def package_dependency_lock(source, output):
    import shutil
    from .bundle_io import sha256_file
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise ValueError("dependency lock file is required for tracking")
    destination = output / "uv.lock"
    if destination.exists():
        raise FileExistsError("dependency lock output already exists")
    shutil.copyfile(source, destination)
    return {"filename": "uv.lock", "sha256": sha256_file(destination), "bytes": destination.stat().st_size}


def verify_dependency_lock(directory, metadata):
    from .bundle_io import sha256_file
    if metadata.get("filename") != "uv.lock":
        raise ValueError("invalid dependency lock filename")
    path = Path(directory) / "uv.lock"
    if sha256_file(path) != metadata.get("sha256") or path.stat().st_size != metadata.get("bytes"):
        raise ValueError("dependency lock provenance mismatch")
    return True


def verify_fresh_process(model_uri, tracking_uri, questions, expected, output, timeout=180):
    import json
    import os
    import subprocess
    import sys
    import tempfile
    script = """
import json, sys
from pathlib import Path
import mlflow
import pandas as pd
mlflow.set_tracking_uri(sys.argv[1])
model = mlflow.pyfunc.load_model(sys.argv[2])
frame = pd.DataFrame(json.loads(Path(sys.argv[3]).read_text()))
observed = model.predict(frame)
from retrieval_lab.tracking import json_records
import retrieval_lab
Path(sys.argv[4]).write_text(json.dumps({"predictions": json_records(observed), "module": retrieval_lab.__file__}))
"""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
    with tempfile.TemporaryDirectory(prefix="fresh-parity-", dir=output) as directory:
        request, result = Path(directory) / "request.json", Path(directory) / "result.json"
        request.write_text(json.dumps(questions.to_dict(orient="records")))
        completed = subprocess.run([sys.executable, "-I", "-c", script, tracking_uri, model_uri, str(request), str(result)],
                                   cwd=directory, env=env, capture_output=True, text=True, timeout=timeout)
        if completed.returncode:
            raise ValueError(f"fresh-process reload failed: {completed.stderr[-2000:]}")
        observed = json.loads(result.read_text())
    if observed["predictions"] != expected:
        raise ValueError("fresh-process prediction parity mismatch")
    module = Path(observed["module"]).resolve()
    if module.parent == Path(__file__).resolve().parent:
        raise ValueError("fresh-process reload imported checkout code")
    return {"prediction_parity": True, "offline": True, "isolated_python": True, "module": str(module)}
