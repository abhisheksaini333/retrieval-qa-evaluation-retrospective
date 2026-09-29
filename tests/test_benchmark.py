from pathlib import Path
import json
import subprocess
import sys

from retrieval_lab.benchmark import audit_data


def test_committed_data_is_disjoint_and_all_answer_spans_exist():
    audit = audit_data(Path("data"))
    assert audit["document_count"] == 24
    assert audit["dev_count"] == audit["test_count"] == 12
    assert audit["gold_doc_overlap"] == []
    assert len(audit["corpus_sha256"]) == 64


def test_cli_audit_produces_machine_readable_output():
    output = subprocess.check_output([sys.executable, "-m", "retrieval_lab.benchmark", "audit"], text=True)
    assert json.loads(output)["document_count"] == 24
