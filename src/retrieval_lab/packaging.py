"""Validate a built wheel outside the checkout and without editable imports."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import zipfile


def verify_wheel(wheel):
    wheel = Path(wheel).resolve()
    with zipfile.ZipFile(wheel) as archive:
        entrypoints = [name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt")]
        if len(entrypoints) != 1 or "retrieval-eval = retrieval_lab.benchmark:main" not in archive.read(entrypoints[0]).decode():
            raise ValueError("wheel is missing the retrieval-eval entry point")
        if "retrieval_lab/benchmark.py" not in archive.namelist():
            raise ValueError("wheel is missing executable package code")
        for member in archive.namelist():
            if member.startswith("/") or ".." in Path(member).parts or "\\" in member:
                raise ValueError("unsafe wheel member")
        with tempfile.TemporaryDirectory(prefix="wheel-proof-", dir=wheel.parent) as directory:
            site = Path(directory) / "site"
            archive.extractall(site)
            code = "import sys; sys.path.insert(0, sys.argv.pop(1)); from retrieval_lab.benchmark import main; main()"
            result = subprocess.run([sys.executable, "-I", "-c", code, str(site), "--help"],
                                    cwd=directory, capture_output=True, text=True, timeout=30)
            if result.returncode or "predict-batch" not in result.stdout or "inspect" not in result.stdout:
                raise ValueError(f"installed wheel CLI failed: {result.stderr}")
            proof = "import sys,json; sys.path.insert(0,sys.argv[1]); import retrieval_lab; from retrieval_lab.core import BM25,Document; print(json.dumps({'module':retrieval_lab.__file__,'ranked':BM25([Document('a','Title','kitten cat')]).search('cat',1)}))"
            executed = subprocess.run([sys.executable, "-I", "-c", proof, str(site)], cwd=directory,
                                      capture_output=True, text=True, timeout=30)
            if executed.returncode:
                raise ValueError(f"installed wheel inference failed: {executed.stderr}")
            record = json.loads(executed.stdout)
            if not Path(record["module"]).is_relative_to(site) or record["ranked"][0][0] != "a":
                raise ValueError("wheel imported code outside the installed package")
    return {"wheel": wheel.name, "isolated_import": True, "cli_help": True, "bm25_prediction": True}
