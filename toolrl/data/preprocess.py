"""Read generated training records without changing their prompt or target."""
import hashlib
import json
import random
from pathlib import Path
from ..rewards import calls, parse


def digest(v):
    return hashlib.sha256(json.dumps(v, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def user_keys(prompt):
    return {digest(" ".join(m["content"].casefold().split())) for m in prompt if m["role"]=="user" and m["content"].strip()}

def query_key(prompt):
    user = [m['content'] for m in prompt if m['role'] == 'user' and m['content'].strip()]
    if not user: raise ValueError('missing user message')
    return digest(' '.join(user[-1].casefold().split()))


def rl_rows(path):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=['prompt','reward_model','extra_info']):
        for r in batch.to_pylist():
            try:
                prompt = r['prompt']
                if not isinstance(prompt,list) or not all(isinstance(m,dict) and isinstance(m.get('content'),str) and isinstance(m.get('role'),str) for m in prompt): continue
                truth = calls(r['reward_model']['ground_truth'])
                yield dict(id=digest(prompt), query_id=query_key(prompt), group_id=r['extra_info']['group_id'], prompt=prompt,
                           answer=json.dumps(truth,ensure_ascii=False,sort_keys=True,allow_nan=False))
            except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError): continue


def sft_rows(path):
    for r in json.loads(Path(path).read_text(encoding='utf-8')):
        try:
            text=r['output'].strip()
            # Preserve the supervised target, remove any text following the first complete call.
            end=text.index('</tool_call>')+len('</tool_call>')
            text=text[:end]
            truth=parse(text,strict=True)
            user='\n'.join(x for x in [r.get('instruction',''),r['query']] if x)
            prompt=[dict(role='system',content=r['system_prompt']),dict(role='user',content=user)]
            yield dict(id=digest(prompt),query_id=query_key(prompt),group_id=r['group_id'],prompt=prompt,
                       answer=json.dumps(truth,ensure_ascii=False,sort_keys=True,allow_nan=False),target=text)
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError): continue


def select(source,tok,n,max_prompt,max_response,seed,blocked):
    # Shuffle generated records first, tokenize only as many candidates as needed.
    source=list(source); random.Random(seed).shuffle(source)
    result=[]; used=set(blocked)
    if n == 0: return result
    for row in source:
        if row.get('group_id',row['query_id']) in used or row['query_id'] in used: continue
        try:
            ids=tok.apply_chat_template(row['prompt'],add_generation_prompt=True,tokenize=True)
            length=len(ids)
            target=row.get('target',row['answer'])
            budget=len(tok.encode(target,add_special_tokens=False))+(1 if 'target' in row else 48)
            if length>max_prompt or budget>max_response: continue
            truth=calls(row['answer'])
            result.append(dict(row,prompt_ids=list(ids),prompt_tokens=length,
                               difficulty=length+32*len(truth)+8*sum(len(c['arguments']) for c in truth)))
            used.update((row['query_id'],row.get('group_id',row['query_id'])))
            if len(result)==n: return result
        except (ValueError,TypeError,KeyError,AttributeError,OverflowError): continue
    raise RuntimeError(f'Only {len(result)}/{n} eligible records; increase length caps or reduce sample counts.')


def build_splits(train,test,sft,tok,*,sft_n,dpo_n,pool_n,val_n,test_n,max_prompt,max_response,sft_response,seed):
    train=list(train); test=list(test)
    def keys(rows): return {k for r in rows for k in (r['query_id'],r.get('group_id',r['query_id']))}
    test_keys=keys(test)
    te=select(test,tok,test_n,max_prompt,max_response,seed+1,set())
    va=select(train,tok,val_n,max_prompt,max_response,seed+2,test_keys)
    blocked=test_keys|keys(va)
    supervised=select(sft,tok,sft_n,max_prompt,sft_response,seed+3,blocked)
    blocked.update(keys(supervised))
    dpo=select(train,tok,dpo_n,max_prompt,max_response,seed+4,blocked)
    blocked.update(keys(dpo))
    pool=select(train,tok,pool_n,max_prompt,max_response,seed+5,blocked)
    return dict(train=supervised,dpo=dpo,grpo_pool=pool,val=va,test=te)
