"""Compare serial and batched generation under the same checkpoint and token cap."""
import argparse,json,time,statistics
from pathlib import Path
from ..train import load,rows,prompt_ids
from ..trainers.rollout import generate_group,synchronize
from ..rewards import structured_score


def main():
    p=argparse.ArgumentParser()
    for k in ('model','data','out'): p.add_argument('--'+k,required=True)
    p.add_argument('--device',default='mps',choices=['mps','cpu','cuda']);p.add_argument('--dtype',default='float32',choices=['float32','float16','bfloat16'])
    p.add_argument('--limit',type=int,default=8);p.add_argument('--group',type=int,default=4);p.add_argument('--batch',type=int,default=2);p.add_argument('--max-response',type=int,default=256);p.add_argument('--temperature',type=float,default=.7);p.add_argument('--seed',type=int,default=42);p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args()
    if min(a.limit,a.group,a.batch,a.max_response,a.repeats)<1:p.error('positive sizes required')
    from transformers import set_seed
    torch,device,tok,model=load(a); data=rows(a.data)[:a.limit]
    if not data: raise ValueError('empty dataset')
    prompts=[prompt_ids(tok,r,device) for r in data]; records=[]
    for repeat in range(a.repeats):
        for batch in ([1,a.batch] if repeat%2==0 else [a.batch,1]):
            a.rollout_batch=batch;set_seed(a.seed+repeat)
            # Warm-up outside measured region, then reset seed.
            list(generate_group(model,tok,prompts[0],a,a.group,True));synchronize(device);set_seed(a.seed+repeat)
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            n=correct=0;start=time.perf_counter()
            for row,ids in zip(data,prompts):
                for _,text,tokens in generate_group(model,tok,ids,a,a.group,True):
                    n+=tokens;correct+=structured_score(text,row['answer'])
            synchronize(device);seconds=time.perf_counter()-start
            record=dict(repeat=repeat,batch=batch,prompts=len(data),rollouts=len(data)*a.group,tokens=n,seconds=seconds,tokens_per_second=n/seconds,rollouts_per_second=len(data)*a.group/seconds,mean_reward=correct/(len(data)*a.group),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
            records.append(record);print(json.dumps(record),flush=True)
    medians={str(b):statistics.median(r['tokens_per_second'] for r in records if r['batch']==b) for b in {1,a.batch}}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(dict(config=vars(a),records=records,median_tokens_per_second=medians,speedup=medians[str(a.batch)]/medians['1']),indent=2))

if __name__=='__main__':main()
