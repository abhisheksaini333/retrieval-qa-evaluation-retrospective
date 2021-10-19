"""Local MLflow tracking and actual pyfunc serialization/reload parity."""
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


def log_and_reload(bundle, output, metrics, questions, artifact_files=()):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle_path = output / "bundle"
    bundle.save(bundle_path)
    expected = prediction_frame([bundle.predict(q) for q in questions["question"]], index=questions.index)
    mlflow.set_tracking_uri((output / "mlruns").as_uri())
    mlflow.set_experiment("retrieval-qa-evaluation")
    with mlflow.start_run(run_name="dense-qa-cpu") as run:
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
            artifacts={"bundle": str(bundle_path)},
            code_paths=[str(Path(__file__).resolve().parent)],
            signature=prediction_signature(), input_example=questions,
            pip_requirements=["numpy==1.26.4", "pandas==2.2.3", "torch==2.5.1",
                              "transformers==4.44.2", "huggingface-hub==0.26.2", "mlflow==2.17.2"],
        )
        loaded = mlflow.pyfunc.load_model(model.model_uri)
        observed = loaded.predict(questions)
        pd.testing.assert_frame_equal(expected, observed, check_exact=True)
        mlflow.log_metric("reload_prediction_parity", 1.0)
        return {"tracking_uri": mlflow.get_tracking_uri(), "run_id": run.info.run_id,
                "model_uri": model.model_uri, "prediction_parity": True,
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
