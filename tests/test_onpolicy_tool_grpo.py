"""On-policy group-relative action policy must use matching full-softmax logits."""
import json
import pytest

from scripts.train_onpolicy_tool_grpo import (
    select_train,masked_action_logprobs,clipped_loss,
)


def test_selection_uses_train_id_only_and_is_order_invariant():
    from sidra.schema import Question
    qs=[Question(f"train_{i:03d}","question","answer",split="train")
        for i in range(20)]
    x=select_train(qs,{"sealed_dev"},seed=43,max_questions=8)
    y=select_train(list(reversed(qs)),{"sealed_dev"},seed=43,max_questions=8)
    assert [q.question_id for q in x]==[q.question_id for q in y]
    with pytest.raises(ValueError,match="TRAIN"):
        select_train(qs,{"train_000"},seed=43,max_questions=8)
    with pytest.raises(ValueError,match="TRAIN"):
        select_train([Question("dev","question","answer",split="dev")],set(),
                     seed=43,max_questions=1)


def test_only_assistant_tokens_have_gradient():
    torch=pytest.importorskip("torch")
    class ToyModel:
        def __call__(self,input_ids):
            self.logits=torch.zeros(1,input_ids.shape[-1],8,requires_grad=True)
            return type("R",(),{"logits":self.logits})()
    m=ToyModel()
    ids=torch.tensor([[1,2,3,4,5]])
    lp=masked_action_logprobs(m,ids,context_length=3)
    assert len(lp)==2
    (-lp.sum()).backward()
    assert m.logits.grad[0,0].abs().sum().item()==0
    assert m.logits.grad[0,1].abs().sum().item()==0
    assert m.logits.grad[0,2].abs().sum().item()>0
    assert m.logits.grad[0,3].abs().sum().item()>0
    assert m.logits.grad[0,4].abs().sum().item()==0


def test_clipped_loss_at_current_policy_matches_group_advantage():
    torch=pytest.importorskip("torch")
    old=torch.tensor([-1.2,-0.6])
    new=old.clone().requires_grad_()
    loss=clipped_loss(new,old,adv=2.0,epsilon=0.2)
    assert loss.item()==pytest.approx(-2.0)
    loss.backward()
    assert new.grad is not None and torch.isfinite(new.grad).all()
    assert new.grad.abs().sum().item()>0


def test_loss_rejects_invalid_clipping():
    torch=pytest.importorskip("torch")
    with pytest.raises(ValueError):clipped_loss(torch.tensor([0.]),torch.tensor([0.]),0,epsilon=0)
