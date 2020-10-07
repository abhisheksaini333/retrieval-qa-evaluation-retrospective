import json
from pathlib import Path
import pytest



def test_strict_json_records(tmp_path):
    from retrieval_lab.core import _jsonl
    for text in ['{"id":1,"id":2}', '{"id":NaN}', '{"id":Infinity}']:
        path = tmp_path / 'rows.jsonl'; path.write_text(text)
        with pytest.raises(ValueError): _jsonl(path)



def test_jsonl_resource_limits(tmp_path):
    from retrieval_lab.core import _jsonl
    path = tmp_path / 'rows.jsonl'; path.write_text('{"a":1}\n{"a":2}\n')
    assert len(_jsonl(path, max_bytes=100, max_records=2)) == 2
    with pytest.raises(ValueError, match='byte'): _jsonl(path, max_bytes=4)
    with pytest.raises(ValueError, match='record'): _jsonl(path, max_records=1)

