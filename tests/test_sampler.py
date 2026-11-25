from toolrl.sampling import RolloutSampler, BUCKETS
from toolrl.train import audit_rollout
import pytest

ROWS=[dict(id=str(i),difficulty=i) for i in range(30)]

@pytest.mark.parametrize('adaptive',[False,True])
def test_budget_and_no_replacement(adaptive):
    s=RolloutSampler(ROWS,12,42,adaptive)
    ids=[]
    for _ in range(12):
        row,k=s.draw(); ids.append(row['id']); s.observe(k,1)
    assert len(set(ids))==12
    with pytest.raises(StopIteration): s.draw()

def test_uniform_seed():
    a=RolloutSampler(ROWS,12,42); b=RolloutSampler(ROWS,12,42)
    assert [a.draw() for _ in range(12)]==[b.draw() for _ in range(12)]

def test_warmup():
    s=RolloutSampler(ROWS,12,42,True)
    for _ in range(6):
        _,k=s.draw(); s.observe(k,1)
    assert s.n==dict.fromkeys(BUCKETS,2)

def test_adaptation_exploration():
    s=RolloutSampler(ROWS,12,42,True,warmup=0)
    for _ in range(20):
        s.observe('easy',0); s.observe('medium',1); s.observe('hard',0)
    p=s.probabilities()
    assert sum(p.values())==pytest.approx(1)
    assert p['medium']>p['easy'] and min(p.values())>=.2/3

def test_exhaustion():
    s=RolloutSampler(ROWS[:2],2,42,True)
    for _ in range(2):
        _,k=s.draw(); s.observe(k,1)
    assert not s.probabilities()

def test_invalid_budget():
    with pytest.raises(ValueError): RolloutSampler(ROWS,31,42)

def test_audit_same_text():
    truth='[{"name":"f","arguments":{"x":1}}]'
    text='<think>x</think><tool_call>[{"name":"f","arguments":{"x":1,"x":1}}]</tool_call>'
    a=audit_rollout(text,truth)
    assert a['original_reward']==1 and a['sar_reward']==0
    assert 'duplicate key' in a['disagreement_reason']


def test_pipeline_shared_pool_and_parent(tmp_path, monkeypatch):
    """Mock workers only: exercise real split, config, orchestration and report."""
    import csv, json, sys
    from pathlib import Path
    import toolrl.pipeline as p
    root=tmp_path/'run'; invocations=[]
    def fake_run(args):
        args=list(map(str,args))
        def arg(key): return args[args.index(key)+1]
        out=Path(arg('--out')); out.mkdir(parents=True,exist_ok=True)
        if args[0]=='toolrl.prepare':
            n=int(arg('--train'))
            (out/'train.jsonl').write_text(''.join(json.dumps(dict(id=str(i)))+'\n' for i in range(n)))
            (out/'manifest.json').write_text(json.dumps(dict(train=n,train_ids='combined')))
            (out/'grpo_pool.jsonl').write_text(''.join(json.dumps(dict(id=str(i+n)))+'\n' for i in range(int(arg('--pool')))))
            for split in ('val','test'): (out/f'{split}.jsonl').write_text('{}\n')
        elif args[1]=='eval':
            (out/'metrics.json').write_text(json.dumps(dict(n=1,test_id_hash='fixed',max_response=128,exact_match=0)))
        elif args[1].startswith('grpo'):
            invocations.append((args[1],arg('--model'),arg('--data'),arg('--limit')))
            (out/'training.json').write_text(json.dumps(dict(steps=4,rollouts=16)))
    monkeypatch.setattr(p,'run',fake_run)
    dataset=tmp_path/'dataset';dataset.mkdir()
    for name in ('train.parquet','test.parquet','sft.json'):(dataset/name).write_text('fixture')
    (dataset/'manifest.json').write_text(json.dumps({'output_hashes':{name:p.checksum(dataset/name) for name in ('train.parquet','test.parquet','sft.json')}}))
    monkeypatch.setattr(p,'resolve_model',lambda *a:(tmp_path/'model',{'model':'fixture','revision':'test'}))
    monkeypatch.setattr(sys,'argv',['pipeline','--run',str(root),'--data-dir',str(dataset),'--sft','8','--grpo','4','--grpo-pool','12','--skip-dpo'])
    p.main()
    assert len(invocations)==3
    assert {x[1:] for x in invocations}=={(str(root/'E1'/'merged'),str(root/'data'/'grpo_pool.jsonl'),'4')}
    ids=lambda file:{json.loads(x)['id'] for x in file.read_text().splitlines()}
    s=ids(root/'data'/'train.jsonl'); g=ids(root/'data'/'grpo_pool.jsonl')
    assert len(s)==8 and len(g)==12 and not s&g
    with (root/'report'/'summary.csv').open() as f: result=list(csv.DictReader(f))
    assert [r['stage'] for r in result]==['E0','E1','E3','E4','E5']
    assert result[-1]['parent']=='E1' and result[-1]['rollouts']=='16'
