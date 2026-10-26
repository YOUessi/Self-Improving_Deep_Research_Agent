"""No DEV leakage, assistant-only masks and nonzero relative policy gradients."""
import pytest

from scripts.replay_masked_grpo_pilot import validate_train_replay


def fake_rollout(qid,ix,adv):
    return {"question_id":qid,"rollout_index":ix,
            "group_advantage":adv,
            "trace":{"question_id":qid,"termination":"finish"},
            "model_action_io":[{
                "messages":[{"role":"system","content":"Search, read and cite"},
                            {"role":"user","content":'{"question":"What?","tool_history":[]}'}],
                "raw_completion":'{"kind":"search","query":"A"}',
            }]}


def test_replay_groups_check_label_isolation(tmp_path):
    import json
    mf=tmp_path/"dev.json"
    mf.write_text(json.dumps({"pilot":{"selected_ids":["DEVID"]},
                              "heldout":{"selected_ids":["TESTID"]}}))
    rows=[fake_rollout("train-a",0,-1.0),fake_rollout("train-a",1,1.0)]
    selected=validate_train_replay(rows,mf)
    assert len(selected)==1 and len(selected[0])==2
    with pytest.raises(ValueError,match="dev"):
        validate_train_replay(rows+[fake_rollout("DEVID",0,0)],mf)
    with pytest.raises(ValueError,match="ground truth"):
        bad=fake_rollout("train-a",0,1)
        bad["model_action_io"][0]["messages"][1]["content"]='{"gold_answer":"x"}'
        validate_train_replay([bad,bad],mf)


def test_clipped_group_loss_only_generates_gradient_for_action_logprobs():
    torch=pytest.importorskip("torch")
    from scripts.replay_masked_grpo_pilot import clip_policy_loss
    old=torch.tensor([-.7,-.5])
    new=torch.tensor([-.7,-.5],requires_grad=True)
    good=clip_policy_loss(new,old,2.0,epsilon=0.2)
    assert float(good.detach()) == pytest.approx(-2.0,abs=1e-5)
    good.backward()
    assert torch.isfinite(new.grad).all()
    assert all(x < 0 for x in new.grad)


def test_assistant_token_mask_excludes_tool_context_logits():
    torch=pytest.importorskip("torch")
    from scripts.replay_masked_grpo_pilot import assistant_token_logprobs
    class FakeModel:
        def __call__(self,input_ids):
            # 4 positions, vocab=8. The completion labels are last two tokens.
            scores=torch.zeros((1,input_ids.shape[1],8),requires_grad=True)
            self.scores=scores
            return type("Output",(),{"logits":scores})
    m=FakeModel()
    tokens=torch.tensor([[1,2,3,4]])
    loss=-assistant_token_logprobs(m,tokens,context_length=2).sum()
    loss.backward()
    # To predict tokens 3 and 4, gradients touch only positions 1 and 2.
    assert torch.equal(m.scores.grad[0,0],torch.zeros(8))
    assert m.scores.grad[0,1].abs().sum()>0
    assert m.scores.grad[0,2].abs().sum()>0
    assert torch.equal(m.scores.grad[0,3],torch.zeros(8))


def test_replay_question_membership_requires_actual_train_dataset(tmp_path):
    import json
    from sidra.io import save_jsonl
    from sidra.schema import Document,Question
    from scripts.replay_masked_grpo_pilot import validate_train_membership
    data=tmp_path/"train";data.mkdir()
    save_jsonl(data/"documents.jsonl",[Document("d","D","context").__dict__])
    save_jsonl(data/"questions.jsonl",[Question("train-a","What?","answer",
                                                    split="train").__dict__])
    ok=fake_rollout("train-a",0,1.0)
    ok["trace"]["question"]="What?"
    digest=validate_train_membership([[ok]],data)
    assert len(digest)==64
    missing=fake_rollout("not-from-train",0,1.0)
    missing["trace"]["question"]="Other"
    with pytest.raises(ValueError,match="TRAIN"):
        validate_train_membership([[missing]],data)
    changed=fake_rollout("train-a",0,1.0)
    changed["trace"]["question"]="Injected fake question"
    with pytest.raises(ValueError,match="TRAIN"):
        validate_train_membership([[changed]],data)
