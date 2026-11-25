import json
from toolrl.train import gold, preference
from toolrl.rewards import structured_score
from toolrl.pipeline import done

ROW={'id':'1','prompt':[{'role':'user','content':'weather'}],'answer':json.dumps([{'name':'weather','arguments':{'city':'Nanjing'}}])}
def test_gold(): assert structured_score(gold(ROW),ROW['answer'])==1
def test_preference_gold_fallback():
    p=preference(ROW,['wrong']); assert p['chosen_source']=='gold' and p['rejected']=='wrong'
def test_preference_real():
    p=preference(ROW,[gold(ROW),'wrong']); assert p['chosen_source']=='sampled'
def test_no_negative(): assert preference(ROW,[gold(ROW)]) is None
def test_done(tmp_path):
    d=tmp_path/'stage'; count=[]; done(d,lambda p:count.append(1)); done(d,lambda p:count.append(2)); assert count==[1]

def test_logp_alignment():
    import pytest
    torch=pytest.importorskip('torch')
    from types import SimpleNamespace
    from toolrl.train import logp
    class Mock(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.values=torch.nn.Parameter(torch.arange(20,dtype=torch.float32).reshape(1,4,5)/10)
        def forward(self,input_ids,attention_mask,use_cache,logits_to_keep):
            return SimpleNamespace(logits=self.values[:,-logits_to_keep:,:])
    m=Mock(); seq=torch.tensor([[0,1,2,3]])
    result=logp(m,seq,2,1.)
    expected=torch.log_softmax(m.values[:,1:3,:],dim=-1).gather(-1,seq[:,2:].unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(result,expected)
    (-result.mean()).backward(); assert m.values.grad.abs().sum()>0
