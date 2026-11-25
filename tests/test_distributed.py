"""CPU tests for the distributed algorithm; CUDA execution is separately smoke-tested."""
import contextlib
from collections import Counter
from types import SimpleNamespace
import pytest
from toolrl.sampling import RolloutSampler
from toolrl.trainers.distributed import windows, left_pad, completion_batch, log_probs, generate_batch, ordered_draws, observe_global, epoch_order


def test_windows_tail():
    plan=list(windows(13,2,2,2))
    assert [r for r,_ in plan]==[8,5]
    assert [x for _,w in plan for b in w for x in b if x is not None]==list(range(13))
    assert all(len(b)==4 for _,w in plan for b in w)
    assert plan[-1][1][-1]==[12,None,None,None]


def test_windows_full_grpo():
    assert [n for n,_ in windows(2048,2,2,8)]==[32]*64


def test_epoch_order_deterministic():
    data=[dict(id=i,prompt_tokens=(i*7)%13) for i in range(57)]
    assert epoch_order(data,42,8)==epoch_order(data,42,8)
    assert {x['id'] for x in epoch_order(data,42,8)}==set(range(57))


def test_sampler_same_global_state():
    data=[dict(id=str(i),difficulty=i) for i in range(24)]
    a,b=[RolloutSampler(data,12,42,adaptive=True) for _ in range(2)]
    used=set()
    for step in range(3):
        x=ordered_draws(a,2,2);y=ordered_draws(b,2,2);assert x==y
        for r,k in x:
            assert r['id'] not in used;used.add(r['id'])
        for s in (a,b):
            for j,(_,k) in enumerate(x):s.observe(k,j%2)
        assert a.q==b.q and a.probabilities()==b.probabilities()
    assert len(used)==12


def test_batch_logp_matches_individual():
    torch=pytest.importorskip('torch')
    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.w=torch.nn.Parameter(torch.arange(60,dtype=torch.float32).reshape(10,6)/10)
        def forward(self,input_ids,attention_mask,position_ids,use_cache,logits_to_keep):
            # The logits depend on the token's unpadded position and token ID.
            return SimpleNamespace(logits=(self.w[input_ids] + position_ids[...,None]*torch.arange(6)/100)[:,-logits_to_keep:,:])
    p=[[1,2,3],[2]];c=[[4],[1,2,5]];m=Fake()
    batch=completion_batch(p,c,0,'cpu');ids,mask,pos,targ,cmask=batch
    assert ids.tolist()==[[1,2,3,4,0,0],[0,0,2,1,2,5]]
    assert pos.tolist()==[[0,1,2,3,3,3],[0,0,0,1,2,3]]
    lp,mask=log_probs(m,batch,.7)
    assert lp[0,1:].tolist()==[0.,0.]
    combined=-(lp.sum(1)/mask.sum(1)).mean();combined.backward();gradient=m.w.grad.clone();m.w.grad=None
    individual=[]
    for x,y in zip(p,c):
        q,mm=log_probs(m,completion_batch([x],[y],0,'cpu'),.7);individual.append(-q.sum()/mm.sum())
    expected=torch.stack(individual).mean();expected.backward()
    assert torch.allclose(combined,expected,atol=1e-6)
    assert torch.allclose(m.w.grad,gradient,atol=1e-6)


def test_generate_mixed_prompts_and_eos():
    torch=pytest.importorskip('torch')
    class Fake:
        generation_config=SimpleNamespace(eos_token_id=[8,9])
        def eval(self):return self
        def generate(self,input_ids,attention_mask,num_return_sequences,**kw):
            assert input_ids.shape[0]==2
            assert attention_mask.tolist()==[[1,1],[0,1]]
            out=[]
            for i in range(2):
                for j in range(num_return_sequences):out.append(torch.cat([input_ids[i],torch.tensor([i+1,8,0])]))
            return torch.stack(out)
    tok=SimpleNamespace(pad_token_id=0,eos_token_id=8,decode=lambda ids,**kw:str(ids))
    a=SimpleNamespace(max_response=3,rollout_batch=2,temperature=.7)
    result=generate_batch(Fake(),tok,[[1,2],[3]],a,5,'cpu',True)
    assert len(result)==2 and all(len(g)==5 for g in result)
    assert all(ids==[1,8] for ids,_ in result[0]);assert all(ids==[2,8] for ids,_ in result[1])


def _gloo_worker(rank, init_file, result_path):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+init_file,rank=rank,world_size=2)
    try:
        # Verify no_sync + masked final microbatch + exact global mean gradient.
        m=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():m.weight.fill_(1.)
        ddp=DDP(m)
        xs=torch.arange(1.,6.).view(-1,1)
        for real,batches in windows(5,2,2,2):
            ddp.zero_grad(set_to_none=True)
            for bi,slots in enumerate(batches):
                local=slots[rank*2:(rank+1)*2];x=torch.stack([xs[i] if i is not None else xs[0] for i in local]);mask=torch.tensor([i is not None for i in local]).float()
                with (contextlib.nullcontext() if bi==len(batches)-1 else ddp.no_sync()):
                    loss=(ddp(x).square().squeeze(-1)*mask).sum()*2/real;loss.backward()
            assert torch.allclose(m.weight.grad,torch.tensor([[22.]]))
        data=[dict(id=str(i),difficulty=i) for i in range(24)]
        sampler=RolloutSampler(data,12,42,adaptive=True);ids=[]
        for i in range(3):
            global_rows=ordered_draws(sampler,2,2);ids.extend(r['id'] for r,_ in global_rows)
            own=global_rows[rank*2:(rank+1)*2]
            observe_global(sampler,[(k,rank) for _,k in own],dist,2)
        got=[None]*2;dist.all_gather_object(got,(ids,sampler.q,sampler.n))
        assert got[0]==got[1] and len(set(ids))==12
        if rank==0:open(result_path,'w').write('PASS')
    finally:dist.destroy_process_group()


def test_real_two_rank_gloo(tmp_path):
    torch=pytest.importorskip('torch')
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():pytest.skip('Gloo unavailable')
    import torch.multiprocessing as mp
    init=tmp_path/'dist_init';out=tmp_path/'result'
    mp.spawn(_gloo_worker,args=(str(init),str(out)),nprocs=2,join=True)
    assert out.read_text()=='PASS'


def _pipeline_worker(rank, init_file, work):
    """Exercise the actual worker code with a tiny stand-in model on Gloo."""
    import json
    import torch
    import torch.distributed as dist
    from pathlib import Path
    from toolrl.trainers import distributed as d
    from toolrl.train import gold
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+init_file,rank=rank,world_size=2)
    class Tiny(torch.nn.Module):
        generation_config=SimpleNamespace(eos_token_id=7)
        def __init__(self):
            super().__init__();self.adapter=torch.nn.Parameter(torch.zeros(8,8));self.enabled=True
        @contextlib.contextmanager
        def disable_adapter(self):
            self.enabled=False
            try:yield
            finally:self.enabled=True
        def forward(self,input_ids,attention_mask,position_ids,use_cache,logits_to_keep):
            v=self.adapter[input_ids] if self.enabled else torch.zeros((*input_ids.shape,8),device=input_ids.device)
            return SimpleNamespace(logits=v[:,-logits_to_keep:,:])
        def generate(self,input_ids,attention_mask,num_return_sequences,**kwargs):
            out=[]
            for row in input_ids:
                for j in range(num_return_sequences):out.append(torch.cat([row,torch.tensor([1 if j%2==0 else 2,7])]))
            return torch.stack(out)
    truth='[{"name":"weather","arguments":{"city":"Nanjing"}}]'
    correct='<think>Choose tool.</think><tool_call>'+truth+'</tool_call>'
    class Tokenizer:
        pad_token_id=0;eos_token_id=7
        def encode(self,text,**kw):return [1,3] if text!='wrong' else [2]
        def decode(self,ids,**kw):return correct if ids[0]==1 else 'wrong'
    d.load=lambda a,device,training:(Tokenizer(),Tiny())
    def fake_save(model,tok,out):
        torch.save(model.adapter.detach(),out/'adapter.pt')
    d.save_model=fake_save
    root=Path(work)
    args=dict(model='dummy',data=str(root/'rows.jsonl'),nproc=2,rank=8,group=2,limit=5,epochs=1,max_response=8,seed=42,rollout_batch=2,train_batch=2,dpo_batch=2,rl_prompts=1,loss_batch=1,accum=2,rl_accum=2,eval_batch=2,lr=.01,beta=.001,dpo_beta=.1,temperature=.7,attention='sdpa',gradient_checkpointing=False)
    try:
        for stage in ('sft','preferences','dpo','grpo','grpo_improved','grpo_das','eval'):
            out=root/stage
            if rank==0:out.mkdir()
            dist.barrier();cfg=args.copy();cfg['stage']=stage
            if stage=='dpo':cfg['data']=str(root/'preferences'/'pairs.jsonl')
            if stage.startswith('grpo'):cfg['limit']=4
            a=SimpleNamespace(**cfg)
            if stage in ('eval','preferences'):d.infer(a,torch,dist,rank,2,torch.device('cpu'),out)
            else:d.train(a,torch,dist,rank,2,torch.device('cpu'),out)
        if rank==0:
            for st in ('grpo','grpo_improved','grpo_das'):
                obj=json.loads((root/st/'training.json').read_text());assert obj['rollouts']==8 and obj['prompt_groups']==4 and obj['effective_groups']==4
                assert len({r['id'] for r in d.rows(root/st/'rollouts.jsonl')})==4
            assert json.loads((root/'sft'/'training.json').read_text())['samples']==5
            assert json.loads((root/'dpo'/'training.json').read_text())['samples']==5
            assert json.loads((root/'eval'/'metrics.json').read_text())['n']==5
            (root/'completed').write_text('PASS')
    finally:dist.destroy_process_group()


def test_actual_workers_two_rank_cpu(tmp_path):
    import json
    torch=pytest.importorskip('torch')
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():pytest.skip('Gloo unavailable')
    import torch.multiprocessing as mp
    data=[dict(id=str(i),prompt_ids=[1,2]+[3]*(i%3),prompt_tokens=2+i%3,difficulty=i,answer='[{"name":"weather","arguments":{"city":"Nanjing"}}]') for i in range(5)]
    (tmp_path/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in data))
    mp.spawn(_pipeline_worker,args=(str(tmp_path/'init'),str(tmp_path)),nprocs=2,join=True)
    assert (tmp_path/'completed').read_text()=='PASS'
