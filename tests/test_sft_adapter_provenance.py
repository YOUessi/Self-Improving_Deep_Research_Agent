"""Pre/post-training comparison must preserve adapter/source provenance."""
import json

from sidra.experiment import evaluate
from sidra.io import save_jsonl
from sidra.schema import Action, Document, Question


class FixedAdapterPolicy:
    name="transformers_pretrained_agent"
    model_name="/models/Qwen3-0.6B-Base"
    adapter_path="/models/adapter-after-train"
    def decide(self,question,observations,remaining):
        return Action("finish",answer="")


def test_real_dev_adapter_metadata_not_untrained(tmp_path):
    data=tmp_path/"data";data.mkdir()
    save_jsonl(data/"documents.jsonl",[Document("d","D","reference").__dict__])
    save_jsonl(data/"questions.jsonl",[
        Question("test","What?","reference",supporting_doc_ids=("d",),
                 scope="global",split="dev").__dict__])
    output=tmp_path/"result"
    evaluate(data,"dev",FixedAdapterPolicy(),output)
    meta=json.loads((output/"manifest.json").read_text())
    assert meta["stage"]=="pretrained_plus_train_only_sft_adapter_evaluation"
    assert meta["adapter_path"]=="/models/adapter-after-train"
    assert meta["model_id"]=="/models/Qwen3-0.6B-Base"
    assert meta["no_weight_updates_in_this_evaluation"]


def test_synthetic_adapter_evaluation_is_still_smoke(fixture_dir,tmp_path):
    evaluate(fixture_dir,"dev",FixedAdapterPolicy(),tmp_path/"result",limit=1)
    m=json.loads((tmp_path/"result"/"manifest.json").read_text())
    assert m["synthetic_fixture"] is True
    assert m["stage"]=="pretrained_model_on_synthetic_smoke"
