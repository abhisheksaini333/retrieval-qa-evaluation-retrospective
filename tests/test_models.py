import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retrieval_lab.core import Document
from retrieval_lab.models import DenseIndex, Encoder, Reader, QABundle, directory_fingerprint

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_models():
    if os.environ.get("RUN_MODEL_TESTS") != "1":
        pytest.skip("set RUN_MODEL_TESTS=1 after downloading real pinned models")
    path = Path("artifacts/models")
    return Encoder(path / "encoder"), Reader(path / "reader")


def documents():
    return [Document("pets", "Pets", "A kitten is a young cat."),
            Document("retention", "Retention", "The archive keeps receipts for seven years.")]


def test_real_dense_encoder_distinguishes_semantics_and_empty_index(real_models):
    encoder, _ = real_models
    vectors = encoder.encode(["a kitten", "a young cat", "database backups"])
    assert vectors.shape == (3, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]
    assert DenseIndex([], encoder).search("kitten", 2) == []
    assert DenseIndex(documents(), encoder).search("young feline", 1)[0][0] == "pets"


def test_bundle_save_reload_and_corruption_detection(tmp_path, real_models):
    encoder, reader = real_models
    bundle = QABundle(documents(), encoder, reader, threshold=0.0, reader_k=1)
    before = bundle.predict("How long does the archive keep receipts?")
    target = tmp_path / "bundle"
    bundle.save(target)
    loaded = QABundle.load(target)
    after = loaded.predict("How long does the archive keep receipts?")
    assert before == after
    assert after["answer"] == "seven years"
    assert after["document_id"] == "retention"
    manifest = json.loads((target / "manifest.json").read_text())
    manifest["doc_ids"].reverse()
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mismatch"):
        QABundle.load(target)


def test_malformed_predict_rejected(real_models):
    encoder, reader = real_models
    bundle = QABundle(documents(), encoder, reader, threshold=0.5)
    for question in [None, "", "x" * 2049]:
        with pytest.raises(ValueError):
            bundle.predict(question)


def test_fingerprint_detects_weight_bytes(tmp_path):
    p = tmp_path / "model.safetensors"
    p.write_bytes(b"before")
    before = directory_fingerprint(tmp_path)
    p.write_bytes(b"after")
    assert directory_fingerprint(tmp_path) != before


def test_real_mlflow_roundtrip(tmp_path, real_models):
    from retrieval_lab.tracking import log_and_reload
    encoder, reader = real_models
    bundle = QABundle(documents(), encoder, reader, threshold=0.0, reader_k=1)
    questions = pd.DataFrame({"question": ["How long does the archive keep receipts?"]})
    evidence = log_and_reload(bundle, tmp_path, {"test_em": 1.0}, questions)
    assert evidence["prediction_parity"] is True
    assert evidence["run_id"]
    assert evidence["loaded_prediction"][0]["answer"] == "seven years"
