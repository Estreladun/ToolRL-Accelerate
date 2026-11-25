"""Orchestrate a resumable local experiment, one subprocess per stage."""
import argparse, csv, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
from .prepare import digest
from .resources import resolve_model, CATALOG, checksum

STAGES=('prepare','e0','sft','dpo','grpo','improved','das','summary')

def run(args):
    print('>>>', ' '.join(map(str,args)),flush=True)
    subprocess.run([sys.executable,'-m',*map(str,args)],check=True)


def run_distributed(nproc, args):
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={nproc}','-m','--','toolrl.trainers.distributed',*map(str,args)]
    print('>>>', ' '.join(cmd),flush=True)
    subprocess.run(cmd,check=True)


def read_json(path): return json.loads(path.read_text())

def done(path, action):
    mark=path/'.done'
    if mark.is_file():
        print(f'Skip completed: {path}',flush=True); return
    # Remove only this pipeline's incomplete stage directory; prior stages stay intact.
    if path.exists(): shutil.rmtree(path)
    path.mkdir(parents=True)
    action(path)
    mark.touch()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--until',choices=STAGES,default='summary'); p.add_argument('--run',default='runs/mac'); p.add_argument('--model',default='Qwen/Qwen2.5-0.5B-Instruct'); p.add_argument('--device',choices=['mps','cpu','cuda'],default='mps'); p.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='float32'); p.add_argument('--sft',type=int,default=8); p.add_argument('--grpo',type=int,default=4); p.add_argument('--grpo-pool',type=int,default=None); p.add_argument('--dpo',type=int,default=4); p.add_argument('--val',type=int,default=4); p.add_argument('--test',type=int,default=4); p.add_argument('--group',type=int,default=4); p.add_argument('--max-prompt',type=int,default=1024); p.add_argument('--max-response',type=int,default=128); p.add_argument('--rank',type=int,default=8); p.add_argument('--seed',type=int,default=42); p.add_argument('--skip-dpo',action='store_true'); p.add_argument('--data-dir',default='data/processed'); p.add_argument('--sft-response',type=int,default=1024); p.add_argument('--rollout-batch',type=int,default=2); p.add_argument('--model-revision')
    p.add_argument('--backend',choices=['local','ddp'],default='local')
    p.add_argument('--attention',choices=['sdpa','flash_attention_2'],default='sdpa')
    p.add_argument('--gradient-checkpointing',action=argparse.BooleanOptionalAction,default=True)
    for name,value in {'nproc':2,'train-batch':4,'dpo-batch':2,'rl-prompts':2,'loss-batch':2,'accum':8,'rl-accum':8,'eval-batch':4,'sft-epochs':1,'dpo-epochs':1,'rl-epochs':1}.items():p.add_argument('--'+name,type=int,default=value)
    for name,value in {'sft-lr':1e-4,'dpo-lr':5e-6,'rl-lr':5e-6,'beta':.001,'temperature':.7,'dpo-beta':.1}.items():p.add_argument('--'+name,type=float,default=value)
    a=p.parse_args()
    if min(a.sft,a.grpo,a.val,a.test,a.rank,a.rollout_batch,a.sft_response)<1 or a.group<2 or a.dpo<1: p.error('positive sizes and group >= 2 required')
    if min(a.nproc,a.train_batch,a.dpo_batch,a.rl_prompts,a.loss_batch,a.accum,a.rl_accum,a.eval_batch,a.sft_epochs,a.dpo_epochs,a.rl_epochs)<1 or min(a.sft_lr,a.dpo_lr,a.rl_lr,a.temperature,a.dpo_beta)<=0 or a.beta<0:p.error('invalid optimizer or batch settings')
    if a.backend=='ddp' and (a.device!='cuda' or a.dtype!='bfloat16' or a.nproc<2 or a.grpo%(a.nproc*a.rl_prompts)):p.error('DDP requires CUDA/BF16, >=2 GPUs, and GRPO divisible by nproc*rl-prompts')
    if a.grpo_pool is None: a.grpo_pool=3*a.grpo
    if a.grpo_pool<a.grpo: p.error('grpo-pool must be >= grpo')
    root=Path(a.run).resolve(); root.mkdir(parents=True,exist_ok=True)
    directory=Path(a.data_dir).resolve()
    manifest=read_json(directory/'manifest.json')
    sources={}
    for k,name in [('train','train.parquet'),('test','test.parquet'),('sft','sft.json')]:
        path=directory/name
        sha=checksum(path)
        if sha!=manifest['output_hashes'][name]:raise RuntimeError(f'Dataset checksum mismatch: {path}')
        sources[k]=(path,dict(dataset=k,sha256=sha,size=path.stat().st_size))
    dataset_meta={k:v[1] for k,v in sources.items()}
    model_path, model_meta=resolve_model(a.model,a.model_revision)
    cfg={k:v for k,v in vars(a).items() if k not in ('until','run')}; cfg['resources']={'dataset':dataset_meta,'model':model_meta}
    cfg['code_hash']=hashlib.sha256(b''.join(x.read_bytes() for x in sorted(Path(__file__).parent.rglob('*.py')))+CATALOG.read_bytes()).hexdigest()
    conf=root/'config.json'
    if conf.exists() and read_json(conf)!=cfg: raise RuntimeError('Run config changed. Set RUN=runs/new_name to avoid mixing results.')
    conf.write_text(json.dumps(cfg,indent=2))
    data=root/'data'; common=['--device',a.device,'--dtype',a.dtype,'--rank',a.rank,'--max-response',a.max_response,'--group',a.group,'--seed',a.seed,'--rollout-batch',a.rollout_batch]
    def worker(stage,model,source,out,extra=()):
        args=[stage,'--model',model,'--data',source,'--out',out]
        if a.backend=='ddp':
            shared=['--rank',a.rank,'--max-response',a.max_response,'--group',a.group,'--seed',a.seed,'--rollout-batch',a.rollout_batch,'--beta',a.beta,'--temperature',a.temperature,'--dpo-beta',a.dpo_beta]
            for name in ('nproc','attention','train_batch','dpo_batch','rl_prompts','loss_batch','accum','rl_accum','eval_batch'):
                shared.extend(['--'+name.replace('_','-'),getattr(a,name)])
            shared.append('--gradient-checkpointing' if a.gradient_checkpointing else '--no-gradient-checkpointing')
            run_distributed(a.nproc,[*args,*shared,*extra])
        else:
            run(['toolrl.train',*args,*common,'--beta',a.beta,'--temperature',a.temperature,'--dpo-beta',a.dpo_beta,*extra])
    def assess(model,out,split='test'):
        worker('eval',model,data/f'{split}.jsonl',out)
    def prepare(out):
        run(['toolrl.prepare','--train-file',sources['train'][0],'--test-file',sources['test'][0],'--sft-file',sources['sft'][0],
             '--out',out,'--model',model_path,'--train',a.sft,'--dpo',a.dpo,'--pool',a.grpo_pool,'--val',a.val,'--test',a.test,
             '--max-prompt',a.max_prompt,'--max-response',a.max_response,'--sft-response',a.sft_response,'--seed',a.seed])
    def sft(out):
        worker('sft',model_path,data/'train.jsonl',out,['--limit',a.sft,'--lr',a.sft_lr,'--epochs',a.sft_epochs]); assess(out/'merged',out/'eval'); assess(out/'merged',out/'val','val')
    def dpo(out):
        base=root/'E1'/'merged'; decision=dict(use_dpo=False,reason='disabled')
        if not a.skip_dpo:
            worker('preferences',base,data/'dpo.jsonl',out/'preferences',['--limit',a.dpo]); pairs=out/'preferences'/'pairs.jsonl'; stats=read_json(out/'preferences'/'preferences.json')
            if stats['count']:
                worker('dpo',base,pairs,out,['--limit',a.dpo,'--lr',a.dpo_lr,'--epochs',a.dpo_epochs]); assess(out/'merged',out/'eval'); assess(out/'merged',out/'val','val')
                old=read_json(root/'E1'/'val'/'metrics.json')['exact_match']; new=read_json(out/'val'/'metrics.json')['exact_match']
                decision=dict(use_dpo=new>old,reason='validation improved' if new>old else 'no strict validation improvement',sft_val=old,dpo_val=new,pairs=stats['count'])
            else: decision=dict(use_dpo=False,reason='no usable preference pairs')
        (out/'decision.json').write_text(json.dumps(decision,indent=2)); print(decision,flush=True)
    def grpo(out,mode):
        choice=read_json(root/'E2'/'decision.json'); base=root/('E2' if choice['use_dpo'] else 'E1')/'merged'
        worker(mode,base,data/'grpo_pool.jsonl',out,['--limit',a.grpo,'--lr',a.rl_lr,'--epochs',a.rl_epochs]); assess(out/'merged',out/'eval')
    def summary(out):
        decision=read_json(root/'E2'/'decision.json'); records=[]
        for stage in ('E0','E1','E2','E3','E4','E5'):
            path=root/stage/'eval'/'metrics.json'
            if not path.exists():
                if stage=='E2': continue
                raise FileNotFoundError(path)
            rec=read_json(path); rec['stage']=stage; rec['parent']={'E0':'','E1':'E0','E2':'E1','E3':'E2' if decision['use_dpo'] else 'E1','E4':'E2' if decision['use_dpo'] else 'E1','E5':'E2' if decision['use_dpo'] else 'E1'}[stage]; records.append(rec)
        if len({(r['test_id_hash'],r['n'],r['max_response']) for r in records})!=1: raise RuntimeError('Mismatched test sets or generation caps')
        vals={r['stage']:r['exact_match'] for r in records}
        for r in records:
            r['delta_vs_base_pp']=r['exact_match']-vals['E0']; r['delta_vs_parent_pp']=r['exact_match']-vals.get(r['parent'],r['exact_match']); r['delta_vs_grpo_pp']=r['exact_match']-vals['E3'] if r['stage'] in ('E4','E5') else ''
            r['delta_vs_sar_pp']=r['exact_match']-vals['E4'] if r['stage']=='E5' else ''
            stats_path=root/r['stage']/'training.json'
            stats=read_json(stats_path) if r['stage'] in ('E3','E4','E5') else {}
            for key in ('pool_size','steps','prompt_groups','world_size','rollouts','mean_reward','effective_groups','effective_group_ratio','all_zero_groups','all_one_groups','generated_tokens','effective_per_million_tokens','rollout_seconds','rollout_tokens_per_second','effective_groups_per_second','train_seconds','reward_disagreements','original_only','sar_only'):
                r[key]=stats.get(key,'')
        with (out/'summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['stage','parent',*[k for k in records[0] if k not in ('stage','parent')]]); w.writeheader(); w.writerows(records)
        (out/'decision.json').write_text(json.dumps(decision,indent=2)); print((out/'summary.csv').read_text(),flush=True)
    actions=[('prepare',data,prepare),('e0',root/'E0',lambda o:assess(model_path,o/'eval')),('sft',root/'E1',sft),('dpo',root/'E2',dpo),('grpo',root/'E3',lambda o:grpo(o,'grpo')),('improved',root/'E4',lambda o:grpo(o,'grpo_improved')),('das',root/'E5',lambda o:grpo(o,'grpo_das')),('summary',root/'report',summary)]
    for name,path,action in actions:
        done(path,action)
        if name==a.until: break

if __name__=='__main__': main()
