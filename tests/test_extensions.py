import json
from pathlib import Path
import pytest



def test_strict_json_records(tmp_path):
    from retrieval_lab.core import _jsonl
    for text in ['{"id":1,"id":2}', '{"id":NaN}', '{"id":Infinity}']:
        path = tmp_path / 'rows.jsonl'; path.write_text(text)
        with pytest.raises(ValueError): _jsonl(path)

