import json

from sidra.io import load_dataset


def test_hotpotqa_converted_passage_scope(tmp_path):
    from importlib.util import spec_from_file_location, module_from_spec
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "convert_hotpotqa.py"
    spec = spec_from_file_location("hotpot_convert", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    inp, out = tmp_path / "raw.json", tmp_path / "out"
    inp.write_text(json.dumps([{
        "_id": "sample123", "question": "Where was the architect born?",
        "answer": "London", "context": [["Architect", ["She was born in London."]],
                                              ["Noise", ["Unrelated facts."]]],
        "supporting_facts": [["Architect", 0]],
    }]))
    assert module.convert(inp, out, "dev") == (1, 2)
    docs, questions = load_dataset(out)
    assert all(d.scope == "sample123" for d in docs)
    assert questions[0].supporting_doc_ids == ("sample123::0",)
