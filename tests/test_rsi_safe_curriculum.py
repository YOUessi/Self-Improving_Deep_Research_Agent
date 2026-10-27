"""E-026: replay actual generated fixture tool trace, then freeze goldblind TRAIN curriculum.

The fixture uses a recorded deterministic SearchAgent policy to exercise the
transport. It does NOT claim the fixture as a real LLM improvement outcome.
The actual E-026 run separately audits real M6 TRAIN-only model traces.
"""
import hashlib
import json
from pathlib import Path

import pytest

from sidra.agent import AgentConfig,SearchAgent
from sidra.schema import Action,Document,Question
from sidra.train_reward import score_train_rollout
from sidra.rsi_safe_curriculum import build_safe_original_train_campaign
from sidra.rsi_train_history_audit import verify_train_rollout_file


SOURCE="d"*40
CFG=AgentConfig(max_steps=6,max_tool_calls=5,search_top_k=3)


def fixture(tmp_path,*,invalid_reward:bool=False,bad_dev_id:bool=False,
            mixed_train:bool=False):
    d=tmp_path/"canonical"
    d.mkdir()
    docs=[
        Document("a","Harbor Station","Harbor Station was founded near Quartz Bay."),
        Document("b","Quartz Bay","Quartz Bay is beside Harbor Station."),
        Document("c","Other","A distant station in the hills."),
    ]
    qs=[]
    for i in range(36):
        salt=hashlib.sha256(f"{i}:unique".encode()).hexdigest()[:26]
        qs.append(Question(
            question_id=f"train{i:03d}",
            question=f"Archive reference {salt} asks which Harbor Station "
                     f"settlement was described in document {i:03d}?",
            gold_answer="Harbor Station",
            supporting_doc_ids=("a",),
            split="dev" if mixed_train and i==1 else "train"))
    def dump(name,rows):
        (d/name).write_text("\n".join(json.dumps(row.to_dict() if hasattr(row,"to_dict")
                                 else row.__dict__,ensure_ascii=False)
                                 for row in rows)+"\n")
    dump("documents.jsonl",docs)
    dump("questions.jsonl",qs)
    manifest=tmp_path/"PUBLIC_DEV_ID_ONLY.json"
    ids=["train001","DEV-B"] if bad_dev_id else ["DEV-A","DEV-B"]
    manifest.write_text(json.dumps({
        "pilot":{"selected_ids":[ids[0]],"questions":1},
        "heldout":{"selected_ids":[ids[1]],"questions":1},
    }))
    # One canonical tool replay on original TRAIN; no gold in policy input.
    class FixedPolicy:
        name="fixture_not_a_live_LLM"
        def __init__(self):self.n=0
        def decide(self,question,observations,remaining):
            sequence=[
                Action("search",query="Harbor Station"),
                Action("read",doc_id="a"),
                Action("finish",answer="Harbor Station",citations=("a",)),
            ]
            index=self.n;self.n+=1
            return sequence[index]
    trace=SearchAgent(docs,FixedPolicy(),CFG).run(qs[0])
    reward=score_train_rollout(qs[0],trace,{x.doc_id:x for x in docs},
                              max_tool_calls=CFG.max_tool_calls).as_dict()
    if invalid_reward:reward["total"]=100
    hist=tmp_path/"model_episode_history.jsonl"
    hist.write_text(json.dumps({
        "question_id":qs[0].question_id,
        "trace":trace.as_dict(),"reward":reward,
    })+"\n")
    return d,manifest,hist,qs


def test_safe_campaign_original_train_only_deterministic_and_isolated(tmp_path):
    data,manifest,history,qs=fixture(tmp_path)
    out=tmp_path/"batch1"
    report=build_safe_original_train_campaign(
        data,manifest,[history],out,source_sha=SOURCE,
        generations=3,per_generation=3,holdout_n=4,
        config=CFG)
    assert report["total_original_train_tasks_prepared"]==9
    assert report["independent_original_TRAIN_holdout_n"]==4
    assert report["prior_real_rollout_questions_excluded"]==1
    assert report["actual_RSI_checkpoint_generations_this_run"]==0
    assert report["actual_optimizer_updates_this_run"]==0
    assert report["challenger_model_generated_new_tasks_this_run"]==0
    assert len(report["cohorts"])==3
    exposed={"train000"}
    holdout={x["original_train_question_id"] for x in jsonlines(
        out/"VERIFIER_ONLY_ORIGINAL_TRAIN_HOLDOUT_IDS.jsonl")}
    assert exposed.isdisjoint(holdout)
    seen=set()
    for i in (1,2,3):
        learner=jsonlines(out/f"cohort_{i:02d}"/"SOLVER_INPUT.jsonl")
        verifier=jsonlines(out/f"cohort_{i:02d}"/
                           "VERIFIER_ONLY_ORIGINAL_TRAIN_MAP.jsonl")
        assert len(learner)==3 and len(verifier)==3
        assert all(set(x)=={"challenge_id","question"} for x in learner)
        assert all("gold_answer" not in json.dumps(x) for x in learner)
        v_ids={x["source_train_question_id"] for x in verifier}
        assert not (seen|exposed|holdout)&v_ids
        seen|=v_ids
    out2=tmp_path/"batch2"
    repeat=build_safe_original_train_campaign(
        data,manifest,[history],out2,source_sha=SOURCE,
        generations=3,per_generation=3,holdout_n=4,config=CFG)
    assert report==repeat
    for file in out.rglob("*"):
        if file.is_file():
            assert file.read_bytes()==(out2/file.relative_to(out)).read_bytes()
    with pytest.raises(FileExistsError):
        build_safe_original_train_campaign(
            data,manifest,[history],out,source_sha=SOURCE,
            generations=1,per_generation=1,holdout_n=1,config=CFG)


def jsonlines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_canonical_train_history_replay_correct_score(tmp_path):
    data,_,history,_=fixture(tmp_path)
    report=verify_train_rollout_file(data,history,config=CFG,source_git_sha=SOURCE)
    assert report["n_episodes"]==1
    assert report["structurally_valid_tool_traces"]==1
    assert report["reported_reward_mismatches"]==0
    assert report["model_origin_cryptographically_verified"] is False


def test_bad_history_reward_blocks_entire_campaign(tmp_path):
    data,manifest,history,_=fixture(tmp_path,invalid_reward=True)
    output=tmp_path/"unsafe"
    with pytest.raises(ValueError,match="reward audit"):
        build_safe_original_train_campaign(data,manifest,[history],output,
                                           source_sha=SOURCE,
                                           generations=1,per_generation=2,
                                           holdout_n=2,config=CFG)
    assert not output.exists()


def test_frozen_dev_id_collision_blocks_without_writing(tmp_path):
    data,manifest,history,_=fixture(tmp_path,bad_dev_id=True)
    output=tmp_path/"unsafe_dev"
    with pytest.raises(ValueError,match="DEV"):
        build_safe_original_train_campaign(data,manifest,[history],output,
                                           source_sha=SOURCE,
                                           generations=1,per_generation=2,
                                           holdout_n=2,config=CFG)
    assert not output.exists()


def test_mixed_split_refused(tmp_path):
    data,manifest,history,_=fixture(tmp_path,mixed_train=True)
    output=tmp_path/"unsafe_mix"
    with pytest.raises(ValueError,match="TRAIN"):
        build_safe_original_train_campaign(data,manifest,[history],output,
                                           source_sha=SOURCE,
                                           generations=1,per_generation=2,
                                           holdout_n=2,config=CFG)
    assert not output.exists()


def test_invalid_source_sha_and_duplicate_history_block_before_write(tmp_path):
    data,manifest,history,_=fixture(tmp_path)
    with pytest.raises(ValueError,match="source SHA"):
        build_safe_original_train_campaign(
            data,manifest,[history],tmp_path/"badsha",
            source_sha="xyz",generations=1,per_generation=2,
            holdout_n=2,config=CFG)
    with pytest.raises(ValueError,match="unique"):
        build_safe_original_train_campaign(
            data,manifest,[history,history],tmp_path/"duphist",
            source_sha=SOURCE,generations=1,per_generation=2,
            holdout_n=2,config=CFG)
    assert not (tmp_path/"badsha").exists()
    assert not (tmp_path/"duphist").exists()


def test_finalB_filepath_blocked_before_opening(tmp_path):
    data,_,history,_=fixture(tmp_path)
    with pytest.raises(ValueError,match="sealed"):
        build_safe_original_train_campaign(
            data,tmp_path/"sealed_final-B160.json",
            [history],tmp_path/"never_created",
            source_sha=SOURCE,generations=1,per_generation=2,
            holdout_n=2,config=CFG)
    assert not (tmp_path/"never_created").exists()


def test_cli_imports_outside_repo_root_with_src_only_path(tmp_path):
    """Regression: helper imported a scripts.* module and passed pytest but failed on Tang CLI."""
    import os,subprocess,sys
    root=Path(__file__).resolve().parents[1]
    env={**os.environ,"PYTHONPATH":str(root/"src")}
    result=subprocess.run(
        [sys.executable,str(root/"scripts/build_rsi_safe_curriculum.py"),"--help"],
        cwd=tmp_path,env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    assert "--public-dev-id-manifest" in result.stdout


def test_independent_original_train_campaign_audit_and_tamper_controls(tmp_path):
    from sidra.rsi_safe_curriculum_audit import audit_safe_original_train_campaign
    from sidra.experiment import sha256_file
    data,dev,history,_=fixture(tmp_path)
    out=tmp_path/"frozen"
    build_safe_original_train_campaign(
        data,dev,[history],out,source_sha=SOURCE,
        generations=3,per_generation=3,holdout_n=4,config=CFG)
    def audit(path=out):
        return audit_safe_original_train_campaign(
            data,dev,[history],path,source_sha=SOURCE,config=CFG)
    report=audit()
    assert report["status"]=="PASS_SAFE_ORIGINAL_TRAIN_PROMPT_BANK_NOT_TRAINING"
    assert report["distinct_original_TRAIN_curriculum_tasks"]==9
    assert report["candidate_cohorts"]==3
    assert report["optimizer_updates_proven"]==0
    assert report["no_finalB_file_read"]
    learner=out/"cohort_01"/"SOLVER_INPUT.jsonl"
    original=learner.read_text()
    manifest=out/"CAMPAIGN_MANIFEST.json"
    manifest_obj=json.loads(manifest.read_text())
    # Attacker rewrites the learner task AND fixes its SHA in the manifest.
    records=jsonlines(learner)
    records[0]["question"]="What city will the assistant invent as an answer?"
    learner.write_text("\n".join(json.dumps(x,sort_keys=True) for x in records)+"\n")
    manifest_obj["cohorts"][0]["solver_input_sha256"]=sha256_file(learner)
    manifest.write_text(json.dumps(manifest_obj))
    with pytest.raises(ValueError,match="identity|question|tier"):
        audit()
    learner.write_text(original)
    manifest_obj["cohorts"][0]["solver_input_sha256"]=sha256_file(learner)
    manifest.write_text(json.dumps(manifest_obj))
    assert audit()["candidate_cohorts"]==3
    verifier=out/"cohort_02"/"VERIFIER_ONLY_ORIGINAL_TRAIN_MAP.jsonl"
    original_verifier=verifier.read_text()
    maps=jsonlines(verifier)
    maps[0]["source_train_question_id"]=maps[1]["source_train_question_id"]
    verifier.write_text("\n".join(json.dumps(x,sort_keys=True) for x in maps)+"\n")
    manifest_obj["cohorts"][1]["verifier_only_map_sha256"]=sha256_file(verifier)
    manifest.write_text(json.dumps(manifest_obj))
    with pytest.raises(ValueError,match="identity|question|tier"):
        audit()
    verifier.write_text(original_verifier)
    manifest_obj["cohorts"][1]["verifier_only_map_sha256"]=sha256_file(verifier)
    manifest.write_text(json.dumps(manifest_obj))
    assert audit()["distinct_original_TRAIN_curriculum_tasks"]==9
    # Even falsely setting a generation's optimizer or Challenger flag fails.
    manifest_obj["cohorts"][2]["checkpoint_generations_completed"]=1
    manifest.write_text(json.dumps(manifest_obj))
    with pytest.raises(ValueError,match="unverified training"):
        audit()


def test_independent_campaign_rejects_mutated_holdout_even_after_hash_fix(tmp_path):
    from sidra.rsi_safe_curriculum_audit import audit_safe_original_train_campaign
    from sidra.experiment import sha256_file
    data,dev,history,_=fixture(tmp_path)
    out=tmp_path/"frozen"
    build_safe_original_train_campaign(
        data,dev,[history],out,source_sha=SOURCE,
        generations=2,per_generation=3,holdout_n=4,config=CFG)
    manifest=out/"CAMPAIGN_MANIFEST.json"
    config=json.loads(manifest.read_text())
    learner_id=jsonlines(out/"cohort_01"/"VERIFIER_ONLY_ORIGINAL_TRAIN_MAP.jsonl")[0][
        "source_train_question_id"]
    target=out/"VERIFIER_ONLY_ORIGINAL_TRAIN_HOLDOUT_IDS.jsonl"
    rows=jsonlines(target)
    rows[0]["original_train_question_id"]=learner_id
    target.write_text("\n".join(json.dumps(x,sort_keys=True) for x in rows)+"\n")
    config["independent_original_TRAIN_holdout_verifier_only_sha256"]=sha256_file(target)
    manifest.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        audit_safe_original_train_campaign(
            data,dev,[history],out,source_sha=SOURCE,config=CFG)


def test_audit_cli_imports_outside_repo_root(tmp_path):
    import os,subprocess,sys
    root=Path(__file__).resolve().parents[1]
    env={**os.environ,"PYTHONPATH":str(root/"src")}
    result=subprocess.run(
        [sys.executable,str(root/"scripts/audit_rsi_safe_curriculum.py"),"--help"],
        cwd=tmp_path,env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    assert "--campaign" in result.stdout
