from types import SimpleNamespace
import pytest
from toolrl.trainers.rollout import trim_completion,generate_group

def test_eos_inclusive():
    torch=pytest.importorskip('torch')
    seq=torch.tensor([[1,2,3,9,0,0]])
    assert trim_completion(seq,2,[9,8]).tolist()==[[1,2,3,9]]
    assert trim_completion(seq,2,8).tolist()==seq.tolist()

def test_group_exact_budget_and_padding():
    torch=pytest.importorskip('torch')
    class Mock:
        generation_config=SimpleNamespace(eos_token_id=[9,8])
        calls=[]
        def eval(self): pass
        def generate(self,input_ids,num_return_sequences,**kwargs):
            self.calls.append(num_return_sequences)
            return torch.tensor([[1,2,3,9,0]]*num_return_sequences)
    tok=SimpleNamespace(eos_token_id=9,pad_token_id=0,decode=lambda x,**kw:str(x.tolist()))
    a=SimpleNamespace(max_response=5,temperature=.7,rollout_batch=2)
    model=Mock(); outputs=list(generate_group(model,tok,torch.tensor([[1,2]]),a,5))
    assert model.calls==[2,2,1] and len(outputs)==5
    assert all(n==2 and seq.tolist()==[[1,2,3,9]] for seq,_,n in outputs)

def test_single_update_detach_gradient():
    torch=pytest.importorskip('torch')
    x=torch.tensor([-.2,-.8],requires_grad=True)
    adv=torch.tensor([1.,-1.])
    ref=x.detach().clone()+.1
    old=x.detach().clone()
    def loss(ratio):
        d=ref-x
        return (-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)+.001*(d.exp()-d-1)).mean()
    g1=torch.autograd.grad(loss((x-old).exp()),x)[0]
    g2=torch.autograd.grad(loss((x-x.detach()).exp()),x)[0]
    assert torch.allclose(g1,g2)
