"""Small, single-device smoke trainer. No CUDA-specific dependencies."""
import argparse, hashlib, json, random, time
from contextlib import ExitStack
from pathlib import Path
from .rewards import calls, structured_score, baseline_score, evaluate, parse, match
from .sampling import RolloutSampler, BUCKETS
from .trainers.rollout import generate_group, synchronize


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def gold(row):
    # Deterministic format-only SFT target, not a teacher-generated reasoning trace.
    if 'target' in row: return row['target']
    return '<think>I will select the appropriate tool and arguments.</think>\n<tool_call>'+json.dumps(calls(row['answer']),ensure_ascii=False,separators=(',', ':'))+'</tool_call>'


def preference(row, candidates):
    """Require a real incorrect call with a valid format, or a malformed real completion."""
    good = [x for x in candidates if structured_score(x,row['answer']) == 1]
    bad = [x for x in candidates if x.strip() and structured_score(x,row['answer']) == 0]
    if not bad: return None
    # Prefer a structurally valid wrong call. Use the supervised target if no correct rollout exists.
    bad.sort(key=lambda x:evaluate(x,row['answer'])['strict_format'], reverse=True)
    chosen = good[0] if good else gold(row)
    return dict(row,chosen=chosen,rejected=bad[0],chosen_source='sampled' if good else 'gold')


def device_of(torch, name):
    if name=='mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable; use DEVICE=cpu only for diagnostics')
    return torch.device(name)


def load(a, training=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device=device_of(torch,a.device)
    # FP32 by default: avoids fp16 underflow on Mac during smoke testing.
    dtype={'float32':torch.float32,'float16':torch.float16,'bfloat16':torch.bfloat16}[a.dtype]
    if a.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if a.dtype=='bfloat16' and (a.device!='cuda' or not torch.cuda.is_bf16_supported()): raise RuntimeError('BF16 requires a supported CUDA device')
    tok=AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=dtype,attn_implementation='eager',local_files_only=True).to(device)
    if training:
        if getattr(model.config,"attention_dropout",0.0) != 0: raise ValueError("Single-update trainer requires zero attention dropout")
        from peft import LoraConfig,get_peft_model
        model=get_peft_model(model,LoraConfig(r=a.rank,lora_alpha=2*a.rank,lora_dropout=0.0,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
        for p in model.parameters():
            if p.requires_grad: p.data=p.data.float()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.print_trainable_parameters()
    model.eval()
    return torch, device,tok,model


def prompt_ids(tok,row,device):
    import torch
    cached=row.get('prompt_ids')
    if cached is None: cached=tok.apply_chat_template(row['prompt'],tokenize=True,add_generation_prompt=True)
    return torch.tensor([cached],device=device,dtype=torch.long)


def target_seq(tok,ids,text):
    import torch
    suffix=tok.encode(text,add_special_tokens=False)+[tok.eos_token_id]
    return torch.cat([ids,torch.tensor([suffix],device=ids.device,dtype=torch.long)],dim=1)


def generate(model,tok,ids,a,sample):
    return next(generate_group(model,tok,ids,a,1,sample))


def logp(model,seq,prompt_len,temp=1.0):
    import torch
    n=seq.shape[1]-prompt_len
    if n<=0: raise ValueError('empty completion')
    logits=model(input_ids=seq,attention_mask=torch.ones_like(seq),use_cache=False,logits_to_keep=n+1).logits[:,:-1,:].float()/temp
    return torch.log_softmax(logits,dim=-1).gather(-1,seq[:,prompt_len:].unsqueeze(-1)).squeeze(-1)


def save_model(model,tok,out):
    # Save adapters, then merge into a standalone checkpoint for the next stage.
    model.save_pretrained(out/'adapter'); tok.save_pretrained(out/'adapter')
    model=model.to('cpu').merge_and_unload(safe_merge=True)
    model.save_pretrained(out/'merged',safe_serialization=True); tok.save_pretrained(out/'merged')


def audit_rollout(text, truth):
    # Score identical completions with both local reward functions.
    original = baseline_score(text, truth)
    sar = structured_score(text, truth)
    if original not in (0.0, 1.0) or sar not in (0.0, 1.0):
        raise ValueError(f'Expected binary rewards, got {original}, {sar}')
    reason = None
    if original != sar:
        try:
            strict = parse(text, strict=True)
            reason = 'strict_match' if match(strict, calls(truth)) else 'strict_argument_or_call_mismatch'
        except (ValueError, TypeError, KeyError, RecursionError, OverflowError) as exc:
            reason = str(exc)
    return dict(original_reward=original, sar_reward=sar, disagreement_reason=reason,
                **evaluate(text, truth))


def train(a):
    from transformers import set_seed
    set_seed(a.seed); random.seed(a.seed)
    torch,device,tok,model=load(a,True)
    is_grpo = a.stage.startswith('grpo')
    data = rows(a.data)
    if not is_grpo:
        data = data[:a.limit]
    elif a.limit > len(data):
        raise ValueError('GRPO budget exceeds candidate pool')
    if not data: raise ValueError('empty training dataset')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=0.0)
    count=effective=tokens=0
    total_rewards=rollouts=disagreements=original_only=sar_only=0
    all_zero=all_one=0
    bucket_stats = {k: dict(groups=0, effective=0) for k in BUCKETS}
    synchronize(device)
    start = time.perf_counter()
    rollout_seconds=0.0
    with ExitStack() as stack:
        f = stack.enter_context((out/'train.jsonl').open('w'))
        audit = stack.enter_context((out/'rollouts.jsonl').open('w')) if is_grpo else None
        for epoch in range(a.epochs):
            sampler = RolloutSampler(data, a.limit, a.seed+epoch, adaptive=a.stage=='grpo_das') if is_grpo else None
            for i in range(a.limit if is_grpo else len(data)):
                r, bucket = sampler.draw() if is_grpo else (data[i], None)
                group_tokens = 0
                ids=prompt_ids(tok,r,device); plen=ids.shape[1]
                opt.zero_grad(set_to_none=True); model.train(); reward=None; info=None
                if a.stage=='sft':
                    seq=target_seq(tok,ids,gold(r))
                    loss=-logp(model,seq,plen).mean(); loss.backward(); value=float(loss.detach().cpu())
                elif a.stage=='dpo':
                    ch=target_seq(tok,ids,r['chosen']); re=target_seq(tok,ids,r['rejected'])
                    with torch.no_grad(), model.disable_adapter():
                        ref=(logp(model,ch,plen).sum()-logp(model,re,plen).sum()).detach()
                    diff=logp(model,ch,plen).sum()-logp(model,re,plen).sum()
                    loss=-torch.nn.functional.logsigmoid(a.dpo_beta*(diff-ref)); loss.backward(); value=float(loss.detach().cpu())
                else:
                    samples=[]; scores=[]
                    synchronize(device); rollout_start=time.perf_counter()
                    completions=list(generate_group(model,tok,ids,a,a.group,True))
                    synchronize(device); rollout_seconds+=time.perf_counter()-rollout_start
                    for j,(seq,text,n) in enumerate(completions):
                        tokens+=n; group_tokens+=n; samples.append(seq)
                        result=audit_rollout(text,r['answer'])
                        original=result['original_reward']; sar=result['sar_reward']
                        score=original if a.stage=='grpo' else sar
                        scores.append(score); rollouts+=1; total_rewards+=score
                        disagreements+=int(original!=sar)
                        original_only+=int(original==1 and sar==0)
                        sar_only+=int(original==0 and sar==1)
                        audit.write(json.dumps(dict(step=count+1,epoch=epoch,id=r['id'],bucket=bucket,
                            rollout=j,answer=r['answer'],prediction=text,tokens=n,training_reward=score,
                            **result),ensure_ascii=False)+'\n')
                    audit.flush()
                    rs=torch.tensor(scores,device=device,dtype=torch.float32)
                    adv=(rs-rs.mean())/(rs.std(unbiased=True)+1e-6)
                    reward=sum(scores)/len(scores); info=int(max(scores)>min(scores)); effective+=info
                    all_zero+=int(max(scores)==0); all_one+=int(min(scores)==1)
                    sampler.observe(bucket,info)
                    bucket_stats[bucket]['groups']+=1
                    bucket_stats[bucket]['effective']+=info
                    value=0.0
                    # One update/group, all rollouts sampled before any parameter update.
                    # Sequential backward reduces peak memory; only LoRA parameters are updated.
                    for j,seq in enumerate(samples):
                        model.eval()
                        with torch.no_grad():
                            with model.disable_adapter(): ref=logp(model,seq,plen,a.temperature).detach()
                        model.train()
                        cur=logp(model,seq,plen,a.temperature)
                        # Single on-policy update: old policy equals current policy before opt.step.
                        # Dropout is disabled in LoRA; evaluate both in identical mode.
                        ratio=(cur-cur.detach()).exp(); d=ref-cur
                        kl=d.exp()-d-1
                        obj=torch.minimum(ratio*adv[j],ratio.clamp(.8,1.2)*adv[j])
                        loss=(-obj+a.beta*kl).mean()/a.group
                        loss.backward(); value+=float(loss.detach().cpu())
                grad=torch.nn.utils.clip_grad_norm_(params,1.0)
                if not torch.isfinite(grad) or not __import__('math').isfinite(value): raise RuntimeError('Nonfinite loss/gradient; keep FP32 or reduce LR')
                opt.step(); count+=1
                record=dict(stage=a.stage,step=count,epoch=epoch,loss=value,grad_norm=float(grad.detach().cpu()),mean_reward=reward,effective_group=info,generated_tokens=tokens)
                if is_grpo:
                    record.update(id=r['id'], bucket=bucket, group_tokens=group_tokens, scores=scores,
                                  bucket_ema=dict(sampler.q), next_bucket_probs=sampler.probabilities() if sampler.adaptive else None)
                line=json.dumps(record); f.write(line+'\n'); f.flush(); print(line,flush=True)
    synchronize(device)
    elapsed=time.perf_counter()-start
    save_model(model,tok,out)
    stats=dict(stage=a.stage,steps=count,source_model=a.model,effective_group_ratio=effective/count if is_grpo else None,generated_tokens=tokens,train_seconds=elapsed)
    if is_grpo:
        stats.update(pool_size=len(data),rollouts=rollouts,mean_reward=total_rewards/rollouts,
                     rollout_seconds=rollout_seconds,rollout_tokens_per_second=tokens/max(rollout_seconds,1e-9),effective_groups_per_second=effective/max(elapsed,1e-9),
                     effective_groups=effective,all_zero_groups=all_zero,all_one_groups=all_one,
                     effective_per_million_tokens=1e6*effective/max(tokens,1),bucket_stats=bucket_stats,
                     reward_disagreements=disagreements,original_only=original_only,sar_only=sar_only)
    dump(out/'training.json',stats)


def preferences(a):
    from transformers import set_seed
    set_seed(a.seed)
    _,device,tok,model=load(a)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); pairs=[]
    for r in rows(a.data)[:a.limit]:
        ids=prompt_ids(tok,r,device)
        cand=[text for _,text,_ in generate_group(model,tok,ids,a,a.group,True)]
        pair=preference(r,cand)
        if pair and len(tok.encode(pair['chosen'],add_special_tokens=False))+1<=a.max_response and len(tok.encode(pair['rejected'],add_special_tokens=False))+1<=a.max_response: pairs.append(pair)
        print(f'Preference pairs: {len(pairs)}',flush=True)
    with (out/'pairs.jsonl').open('w') as f:
        for r in pairs: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    dump(out/'preferences.json',dict(count=len(pairs),gold_fallback=sum(r['chosen_source']=='gold' for r in pairs)))


def eval_model(a):
    _,device,tok,model=load(a); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    data=rows(a.data); totals=dict(exact_match=0,tool_match=0,valid_call=0,strict_format=0); tokens=0
    if not data: raise ValueError('empty evaluation split')
    with (out/'predictions.jsonl').open('w') as f:
        for i,r in enumerate(data,1):
            ids=prompt_ids(tok,r,device); _,text,n=generate(model,tok,ids,a,False); tokens+=n
            scores=evaluate(text,r['answer'])
            for k,v in scores.items(): totals[k]+=v
            f.write(json.dumps(dict(id=r['id'],answer=r['answer'],prediction=text,**scores),ensure_ascii=False)+'\n'); f.flush()
            print(f'{i}/{len(data)} exact={totals["exact_match"]/i:.3f}',flush=True)
    dump(out/'metrics.json',dict(n=len(data),test_id_hash=hashlib.sha256(json.dumps([r['id'] for r in data]).encode()).hexdigest(),max_response=a.max_response,**{k:100*v/len(data) for k,v in totals.items()},avg_generated_tokens=tokens/len(data)))


def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['sft','dpo','grpo','grpo_improved','grpo_das','preferences','eval']); p.add_argument('--model',required=True); p.add_argument('--data',required=True); p.add_argument('--out',required=True); p.add_argument('--device',default='mps',choices=['mps','cpu','cuda']); p.add_argument('--dtype',default='float32',choices=['float32','float16','bfloat16']); p.add_argument('--rank',type=int,default=8); p.add_argument('--max-response',type=int,default=128); p.add_argument('--group',type=int,default=4); p.add_argument('--limit',type=int,default=8); p.add_argument('--epochs',type=int,default=1); p.add_argument('--lr',type=float,default=5e-6); p.add_argument('--beta',type=float,default=.001); p.add_argument('--dpo-beta',type=float,default=.1); p.add_argument('--temperature',type=float,default=.7); p.add_argument('--seed',type=int,default=42); p.add_argument('--rollout-batch',type=int,default=2); a=p.parse_args()
    if a.group<2 or a.limit<1 or a.epochs<1 or a.rollout_batch<1: p.error('group >=2, limit and epochs >=1')
    {'eval':eval_model,'preferences':preferences}.get(a.stage,train)(a)

if __name__=='__main__': main()
