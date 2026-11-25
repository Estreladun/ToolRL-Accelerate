"""Single-node CUDA DDP post-training. Each rank owns complete GRPO groups.

No cross-device model sharding. The base is frozen, only LoRA gradients are reduced.
A group is sampled by a globally deterministic sampler; reward observations are shared
before the next draw. Generation and optimization use the same on-policy weights.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

from ..train import rows, dump, gold, preference, audit_rollout, save_model
from ..rewards import evaluate
from ..sampling import RolloutSampler, BUCKETS


def windows(n, world, micro, accum):
    """Yield optimizer windows of equally sized rank batches, with masked tail slots.

    A None slot is a dummy (zero contribution), so every rank executes the same
    number of forwards/backwards even with an uneven final batch.
    """
    width = world * micro
    for start in range(0, n, width * accum):
        stop = min(n, start + width * accum)
        batches = []
        for j in range(start, stop, width):
            b = list(range(j, min(j + width, stop)))
            batches.append(b + [None] * (width - len(b)))
        yield stop - start, batches


def left_pad(prompts, pad, device):
    import torch
    if not prompts or any(not p for p in prompts):
        raise ValueError('empty prompt batch')
    width = max(map(len, prompts))
    ids = [[pad] * (width-len(p)) + list(p) for p in prompts]
    mask = [[0] * (width-len(p)) + [1] * len(p) for p in prompts]
    return (torch.tensor(ids, device=device, dtype=torch.long),
            torch.tensor(mask, device=device, dtype=torch.long))


def completion_batch(prompts, completions, pad, device):
    """Left pad prompts to align prediction start; right pad completions and mask."""
    import torch
    if len(prompts) != len(completions) or any(not c for c in completions):
        raise ValueError('nonempty aligned completions required')
    ids, pmask = left_pad(prompts, pad, device)
    width = max(map(len, completions))
    targets = torch.tensor([list(c)+[pad]*(width-len(c)) for c in completions], device=device, dtype=torch.long)
    cmask = torch.tensor([[1]*len(c)+[0]*(width-len(c)) for c in completions], device=device, dtype=torch.long)
    mask = torch.cat((pmask, cmask), dim=1)
    pos = (mask.cumsum(1)-1).clamp_min(0)
    return torch.cat((ids, targets), dim=1), mask, pos, targets, cmask


def log_probs(model, batch, temperature=1.0):
    import torch.nn.functional as F
    ids, mask, pos, targets, cmask = batch
    # Qwen2 supports logits_to_keep. The final logit is unused; only completion
    # positions are materialized, including the last prompt-token prediction.
    logits = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                   use_cache=False, logits_to_keep=targets.shape[1]+1).logits[:, :-1, :]
    logits = logits.float() / temperature
    lp = -F.cross_entropy(logits.transpose(1, 2), targets, reduction='none')
    return lp.masked_fill(~cmask.bool(), 0.0), cmask


def truncate(ids, eos):
    for i, token in enumerate(ids):
        if token in eos:
            return ids[:i+1]
    return ids


def generate_batch(model, tok, prompts, a, count, device, sample=True):
    """Batch distinct prompts and multiple completions in each generate() call."""
    import torch
    ids, mask = left_pad(prompts, tok.pad_token_id, device)
    eos = model.generation_config.eos_token_id
    if eos is None: eos = tok.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    model.eval()
    results = [[] for _ in prompts]
    # Cap parallel returns independently of the number of distinct prompts.
    stride = min(a.rollout_batch, count) if sample else 1
    for start in range(0, count, stride):
        k = min(stride, count-start)
        kwargs = dict(max_new_tokens=a.max_response, do_sample=sample, num_return_sequences=k,
                      use_cache=True, eos_token_id=eos, pad_token_id=tok.pad_token_id)
        if sample: kwargs.update(temperature=a.temperature, top_p=1.0, top_k=0)
        else: kwargs.update(temperature=1.0, top_p=1.0, top_k=None)
        with torch.inference_mode():
            outputs = model.generate(input_ids=ids, attention_mask=mask, **kwargs)
        tail = outputs[:, ids.shape[1]:].tolist()
        for j, seq in enumerate(tail):
            seq = truncate(seq, eos_ids)
            if not seq: raise RuntimeError('empty model completion')
            results[j//k].append((seq, tok.decode(seq, skip_special_tokens=True)))
    if any(len(g) != count for g in results): raise RuntimeError('wrong rollout cardinality')
    return results


def prompt(row, tok):
    if 'prompt_ids' in row: return row['prompt_ids']
    return tok.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True)


def target(tok, text):
    return tok.encode(text, add_special_tokens=False) + [tok.eos_token_id]


def sync_context(ddp, sync):
    return contextlib.nullcontext() if sync else ddp.no_sync()


def setup(a):
    import torch
    import torch.distributed as dist
    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK']); world = int(os.environ['WORLD_SIZE'])
    if world != a.nproc or world < 2: raise RuntimeError('launch with torchrun and the configured GPU count')
    if torch.cuda.device_count() < world: raise RuntimeError('not enough visible CUDA GPUs')
    torch.cuda.set_device(local); device = torch.device('cuda', local)
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('cloud trainer requires native BF16 CUDA GPUs')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    from transformers import set_seed
    set_seed(a.seed + rank); random.seed(a.seed + rank)
    return torch, dist, rank, world, device


def load(a, device, training):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
        attn_implementation=a.attention, local_files_only=True).to(device)
    if training:
        if getattr(model.config, 'attention_dropout', 0.0) != 0:
            raise ValueError('on-policy probability reuse requires zero attention dropout')
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2*a.rank,
            lora_dropout=0.0, target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'], task_type='CAUSAL_LM'))
        for p in model.parameters():
            if p.requires_grad: p.data = p.data.float()
        if a.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.print_trainable_parameters()
    model.eval()
    return tok, model


def epoch_order(data, seed, width):
    """Length-bucket within random windows to avoid globally sorting the curriculum."""
    order = list(range(len(data))); random.Random(seed).shuffle(order)
    block = max(width, width*16)
    for i in range(0, len(order), block):
        order[i:i+block] = sorted(order[i:i+block], key=lambda j: data[j].get('prompt_tokens', 0))
    return [data[i] for i in order]


def ordered_draws(sampler, world, local_count):
    return [sampler.draw() for _ in range(world*local_count)]


def observe_global(sampler, local, dist, world, device=None):
    """All ranks apply identical rank-major observations; CUDA uses a small tensor collective."""
    if device is None:
        gathered = [None]*world
        dist.all_gather_object(gathered, local)
    else:
        import torch
        packed=torch.tensor([[BUCKETS.index(k),v] for k,v in local],device=device,dtype=torch.int64)
        buffers=[torch.empty_like(packed) for _ in range(world)]
        dist.all_gather(buffers,packed)
        gathered=[[(BUCKETS[k],v) for k,v in part] for part in torch.stack(buffers).cpu().tolist()]
    for part in gathered:
        for bucket, effective in part: sampler.observe(bucket, effective)


def synchronize(torch, device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def cosine_scheduler(torch, opt, steps):
    warmup=int(steps*.1)
    def scale(i):
        if i<warmup:return i/max(1,warmup)
        return .5*(1+math.cos(math.pi*min(1.,(i-warmup)/max(1,steps-warmup))))
    return torch.optim.lr_scheduler.LambdaLR(opt,scale)


def write_line(handle, obj):
    handle.write(json.dumps(obj, ensure_ascii=False)+'\n')


def merge_rank_files(out, stem, world, ordered=False):
    paths = [out/f'{stem}.rank{i}.jsonl' for i in range(world)]
    with (out/f'{stem}.jsonl').open('w') as f:
        if ordered:
            records = []
            for p in paths: records.extend(rows(p))
            for r in sorted(records, key=lambda r:r['_index']):
                r.pop('_index'); write_line(f, r)
        else:
            for p in paths:
                with p.open() as src:
                    for line in src: f.write(line)
    for p in paths: p.unlink()


def train(a, torch, dist, rank, world, device, out):
    from torch.nn.parallel import DistributedDataParallel
    tok, model = load(a, device, True)
    ddp = DistributedDataParallel(model, device_ids=[device.index] if device.type=='cuda' else None,
                                  output_device=device.index if device.type=='cuda' else None,
                                  broadcast_buffers=False, gradient_as_bucket_view=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, fused=device.type=='cuda')
    data = rows(a.data); is_rl = a.stage.startswith('grpo')
    if not is_rl: data = data[:a.limit]
    if not data: raise ValueError('empty train set')
    n = a.limit if is_rl else len(data)
    micro = a.rl_prompts if is_rl else (a.dpo_batch if a.stage=='dpo' else a.train_batch)
    accum = a.rl_accum if is_rl else a.accum
    if is_rl and (n>len(data) or n%(world*micro)):
        raise ValueError('GRPO groups per epoch must fit pool and divide nproc*rl_prompts')
    total_steps = math.ceil(n/(world*micro*accum))*a.epochs
    sched = cosine_scheduler(torch,opt,total_steps) if a.stage=='sft' else None
    stats = Counter(); bucket_stats = {k:Counter() for k in BUCKETS}
    step = 0; rollout_seconds = 0.
    synchronize(torch,device); dist.barrier(); start = time.perf_counter()
    with contextlib.ExitStack() as stack:
        log = stack.enter_context((out/'train.jsonl').open('w')) if rank==0 else None
        audit = stack.enter_context((out/f'rollouts.rank{rank}.jsonl').open('w')) if is_rl else None
        for epoch in range(a.epochs):
            if is_rl: sampler = RolloutSampler(data,n,a.seed+epoch,adaptive=a.stage=='grpo_das')
            else: order = epoch_order(data,a.seed+epoch,world*micro)
            for real, batches in windows(n,world,micro,accum):
                opt.zero_grad(set_to_none=True)
                before=stats.copy()
                loss_value = torch.zeros((), device=device)
                # Window-normalized loss; DDP averages over ranks, so multiply by world.
                for bi, indices in enumerate(batches):
                    final_micro = bi==len(batches)-1
                    if is_rl:
                        drawn = ordered_draws(sampler,world,micro)
                        local = drawn[rank*micro:(rank+1)*micro]
                        prompts = [prompt(r,tok) for r,_ in local]
                        synchronize(torch,device); t = time.perf_counter()
                        completions = generate_batch(model,tok,prompts,a,a.group,device,True)
                        synchronize(torch,device); rollout_seconds += time.perf_counter()-t
                        all_p=[]; all_c=[]; all_adv=[]; observations=[]
                        for gi, ((r,bucket), group) in enumerate(zip(local,completions)):
                            scores=[]
                            for j,(ids,text) in enumerate(group):
                                result=audit_rollout(text,r['answer']); orig=result['original_reward']; sar=result['sar_reward']
                                score=orig if a.stage=='grpo' else sar
                                scores.append(score); stats['rollouts']+=1; stats['reward_sum']+=score; stats['tokens']+=len(ids)
                                stats['disagreements']+=int(orig!=sar); stats['original_only']+=int(orig==1 and sar==0); stats['sar_only']+=int(sar==1 and orig==0)
                                write_line(audit,dict(epoch=epoch,step=step+1,rank=rank,id=r['id'],bucket=bucket,rollout=j,
                                    answer=r['answer'],prediction=text,tokens=len(ids),training_reward=score,**result))
                                all_p.append(prompts[gi]); all_c.append(ids)
                            mean=sum(scores)/len(scores); sd=math.sqrt(sum((x-mean)**2 for x in scores)/(len(scores)-1))
                            all_adv.extend((x-mean)/(sd+1e-6) for x in scores)
                            effective=int(max(scores)>min(scores)); observations.append((bucket,effective))
                            stats['groups']+=1;stats['effective']+=effective;stats['all_zero']+=int(max(scores)==0);stats['all_one']+=int(min(scores)==1)
                            bucket_stats[bucket]['groups']+=1;bucket_stats[bucket]['effective']+=effective
                        observe_global(sampler,observations,dist,world,device)
                        # Each chunk has identical rank cardinality; sync only at last backward of window.
                        for j in range(0,len(all_c),a.loss_batch):
                            end=min(j+a.loss_batch,len(all_c)); sync=final_micro and end==len(all_c)
                            batch=completion_batch(all_p[j:end],all_c[j:end],tok.pad_token_id,device)
                            model.eval()
                            with torch.no_grad(), model.disable_adapter(): ref,mask=log_probs(model,batch,a.temperature)
                            model.train()
                            with sync_context(ddp,sync):
                                cur,mask=log_probs(ddp,batch,a.temperature)
                                advantage=torch.tensor(all_adv[j:end],device=device,dtype=torch.float32)[:,None]
                                # Single optimizer update on fresh rollouts. Ratio value is one, gradient is nonzero.
                                ratio=(cur-cur.detach()).exp(); d=ref-cur
                                kl=d.exp()-d-1
                                objective=torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage)
                                seq_loss=((-objective+a.beta*kl)*mask).sum(1)/mask.sum(1)
                                loss=seq_loss.sum()*(world/(real*a.group))
                                loss.backward(); loss_value+=loss.detach()
                    else:
                        slots=indices[rank*micro:(rank+1)*micro]
                        local=[order[i] if i is not None else order[0] for i in slots]
                        weights=torch.tensor([int(i is not None) for i in slots],device=device,dtype=torch.float32)
                        prompts=[prompt(r,tok) for r in local]
                        if a.stage=='sft':
                            batch=completion_batch(prompts,[target(tok,gold(r)) for r in local],tok.pad_token_id,device)
                            model.train()
                            with sync_context(ddp,final_micro):
                                lp,mask=log_probs(ddp,batch); seq_loss=-lp.sum(1)/mask.sum(1)
                                loss=(seq_loss*weights).sum()*(world/real); loss.backward(); loss_value+=loss.detach()
                        else:
                            responses=[target(tok,r['chosen']) for r in local]+[target(tok,r['rejected']) for r in local]
                            batch=completion_batch(prompts+prompts,responses,tok.pad_token_id,device)
                            model.eval()
                            with torch.no_grad(),model.disable_adapter():
                                ref,_=log_probs(model,batch); rch,rre=ref.sum(1).chunk(2); refdiff=rch-rre
                            model.train()
                            with sync_context(ddp,final_micro):
                                cur,_=log_probs(ddp,batch);ch,re=cur.sum(1).chunk(2)
                                seq_loss=-torch.nn.functional.logsigmoid(a.dpo_beta*(ch-re-refdiff))
                                loss=(seq_loss*weights).sum()*(world/real);loss.backward();loss_value+=loss.detach()
                        stats['samples']+=sum(i is not None for i in slots)
                grad=torch.nn.utils.clip_grad_norm_(params,1.0)
                if not torch.isfinite(grad).item() or not torch.isfinite(loss_value).item(): raise RuntimeError('nonfinite gradient/loss')
                opt.step();step+=1
                if sched is not None:sched.step()
                window_keys=('groups','effective','rollouts','reward_sum','tokens','all_zero','all_one','samples')
                progress=torch.cat([loss_value.detach().float().reshape(1),torch.tensor([stats[k]-before[k] for k in window_keys],device=device,dtype=torch.float32)])
                dist.all_reduce(progress);global_loss,*values=progress.cpu().tolist()
                window=dict(zip(window_keys,values))
                if rank==0:
                    record=dict(stage=a.stage,epoch=epoch+1,step=step,steps_total=total_steps,loss=global_loss/world,grad_norm=float(grad),lr=opt.param_groups[0]['lr'])
                    if is_rl: record.update(bucket_ema=dict(sampler.q), next_bucket_probs=sampler.probabilities() if sampler.adaptive else None,
                        groups=int(window['groups']),mean_reward=window['reward_sum']/max(1,window['rollouts']),effective_group_ratio=window['effective']/max(1,window['groups']),
                        generated_tokens=int(window['tokens']),all_zero_groups=int(window['all_zero']),all_one_groups=int(window['all_one']))
                    else:record['samples']=int(window['samples'])
                    if device.type=='cuda':record['peak_cuda_gib']=round(torch.cuda.max_memory_allocated(device)/2**30,2)
                    write_line(log,record);log.flush();print(json.dumps(record),flush=True)
            if audit is not None:audit.flush()
    synchronize(torch,device);dist.barrier();elapsed=time.perf_counter()-start
    gathered=[None]*world;dist.all_gather_object(gathered,(dict(stats),{k:dict(v) for k,v in bucket_stats.items()},rollout_seconds,elapsed))
    # Rank 0 alone saves merged checkpoints. All other ranks wait (no racing writes).
    if rank==0:
        total=Counter();buckets={k:Counter() for k in BUCKETS}
        for st,bs,_,_ in gathered:
            total.update(st)
            for k in BUCKETS:buckets[k].update(bs[k])
        wall=max(g[3] for g in gathered); roll=max(g[2] for g in gathered)
        result=dict(stage=a.stage,steps=step,source_model=a.model,world_size=world,train_seconds=wall,generated_tokens=total['tokens'],samples=total['samples'])
        if is_rl:
            groups=total['groups'];tokens=total['tokens']; effective=total['effective']
            if groups!=a.limit*a.epochs or total['rollouts']!=groups*a.group: raise RuntimeError('global rollout budget mismatch')
            merge_rank_files(out,'rollouts',world)
            result.update(pool_size=len(data),prompt_groups=groups,rollouts=total['rollouts'],mean_reward=total['reward_sum']/total['rollouts'],effective_groups=effective,effective_group_ratio=effective/groups,
                all_zero_groups=total['all_zero'],all_one_groups=total['all_one'],effective_per_million_tokens=1e6*effective/max(tokens,1),rollout_seconds=roll,
                rollout_tokens_per_second=tokens/max(roll,1e-9),effective_groups_per_second=effective/max(wall,1e-9),bucket_stats={k:dict(v) for k,v in buckets.items()},reward_disagreements=total['disagreements'],original_only=total['original_only'],sar_only=total['sar_only'])
        del ddp,opt,sched,params
        save_model(model,tok,out);dump(out/'training.json',result)
    dist.barrier()


def infer(a, torch, dist, rank, world, device, out):
    tok,model=load(a,device,False)
    data=rows(a.data)
    if a.stage=='preferences':data=data[:a.limit]
    if not data:raise ValueError('empty inference set')
    local=list(enumerate(data))[rank::world]
    stem='predictions' if a.stage=='eval' else 'pairs'
    with (out/f'{stem}.rank{rank}.jsonl').open('w') as f:
        # Sort only inference batches; output order restored before metrics.
        local.sort(key=lambda ir:ir[1].get('prompt_tokens',0))
        for j in range(0,len(local),a.eval_batch):
            part=local[j:j+a.eval_batch]; prompts=[prompt(r,tok) for _,r in part]
            groups=generate_batch(model,tok,prompts,a,1 if a.stage=='eval' else a.group,device,a.stage!='eval')
            for (i,r),group in zip(part,groups):
                if a.stage=='eval':
                    ids,text=group[0];write_line(f,dict(_index=i,id=r['id'],answer=r['answer'],prediction=text,tokens=len(ids),**evaluate(text,r['answer'])))
                else:
                    pair=preference(r,[text for _,text in group])
                    if pair and all(len(target(tok,pair[k]))<=a.max_response for k in ('chosen','rejected')):write_line(f,dict(_index=i,**pair))
            if rank==0 and (j//a.eval_batch)%20==0: print(f'{a.stage}: rank0 {j+len(part)}/{len(local)}',flush=True)
    dist.barrier()
    if rank==0:
        merge_rank_files(out,stem,world,ordered=True); records=rows(out/f'{stem}.jsonl')
        if a.stage=='preferences':dump(out/'preferences.json',dict(count=len(records),gold_fallback=sum(r['chosen_source']=='gold' for r in records)))
        else:
            if [r['id'] for r in records]!=[r['id'] for r in data]:raise RuntimeError('evaluation order or cardinality mismatch')
            n=len(records); keys=('exact_match','tool_match','valid_call','strict_format')
            dump(out/'metrics.json',dict(n=n,test_id_hash=hashlib.sha256(json.dumps([r['id'] for r in data]).encode()).hexdigest(),max_response=a.max_response,
                **{k:100*sum(r[k] for r in records)/n for k in keys},avg_generated_tokens=sum(r['tokens'] for r in records)/n))
    dist.barrier()


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['sft','dpo','grpo','grpo_improved','grpo_das','preferences','eval'])
    for key in ('model','data','out'):p.add_argument('--'+key,required=True)
    for key,value in {'nproc':2,'rank':8,'group':5,'limit':8,'epochs':1,'max-response':1024,'seed':42,'rollout-batch':5,'train-batch':4,'dpo-batch':2,'rl-prompts':2,'loss-batch':2,'accum':8,'rl-accum':8,'eval-batch':4}.items():p.add_argument('--'+key,type=int,default=value)
    for key,value in {'lr':1e-6,'beta':.001,'dpo-beta':.1,'temperature':.7}.items():p.add_argument('--'+key,type=float,default=value)
    p.add_argument('--attention',choices=['sdpa','flash_attention_2'],default='sdpa');p.add_argument('--gradient-checkpointing',action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args()
    if min(a.nproc,a.rank,a.limit,a.epochs,a.max_response,a.rollout_batch,a.train_batch,a.dpo_batch,a.rl_prompts,a.loss_batch,a.accum,a.rl_accum,a.eval_batch)<1 or a.group<2 or a.temperature<=0 or a.lr<=0: p.error('invalid training configuration')
    torch,dist,rank,world,device=setup(a);out=Path(a.out)
    try:
        if rank==0:out.mkdir(parents=True,exist_ok=True)
        dist.barrier()
        (infer if a.stage in ('eval','preferences') else train)(a,torch,dist,rank,world,device,out)
    finally:dist.destroy_process_group()

if __name__=='__main__':main()
