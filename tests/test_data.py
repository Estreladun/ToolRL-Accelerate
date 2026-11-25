import json
from toolrl.data.preprocess import build_splits,sft_rows,query_key
from pathlib import Path
import pytest

class Tok:
    def apply_chat_template(self,prompt,**kwargs): return [1,2,3]
    def encode(self,text,**kwargs): return list(range(4))

def record(i):
    prompt=[{'role':'user','content':str(i)}]
    return dict(id=str(i),query_id=query_key(prompt),prompt=prompt,answer='[]')

def test_disjoint_deterministic():
    kwargs=dict(sft_n=8,dpo_n=4,pool_n=12,val_n=4,test_n=4,max_prompt=1024,max_response=128,sft_response=256,seed=42)
    tr=[record(i) for i in range(100)]; te=[record(i) for i in range(95,110)]; sf=[dict(record(i),target='<think>x</think><tool_call>[]</tool_call>') for i in range(20,50)]
    a=build_splits(tr,te,sf,Tok(),**kwargs); b=build_splits(tr,te,sf,Tok(),**kwargs)
    assert a==b
    keys=[r['query_id'] for rows in a.values() for r in rows]
    assert len(keys)==len(set(keys))==32
    assert all(r['prompt_ids']==[1,2,3] for rows in a.values() for r in rows)
    assert not {r['query_id'] for r in te}&{r['query_id'] for k,rows in a.items() if k!='test' for r in rows}

def test_sft_preserves_target(tmp_path):
    target='<think>reason</think><tool_call>[{"name":"f","arguments":{}}]</tool_call>'
    p=tmp_path/'sft.json';p.write_text(json.dumps([dict(system_prompt='original',query='q',instruction='',output=target+' tail',group_id='group')]))
    r=list(sft_rows(p))[0]; assert r['target']==target and r['prompt'][0]['content']=='original'
