"""Build train/test/format-SFT resources from raw datasets only."""
import argparse
import collections
import json
import random
import shutil
import tempfile
from pathlib import Path
from .converter import from_toolace, from_xlam, to_rl_row, arrow_schema
from .quality import iter_rl, inspect, decisions
from .preprocess import digest, user_keys
from ..resources import checksum
from ..rewards import parse

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / 'configs/raw_sources.lock.json'


def fetch_originals(directory, local_only=False):
    root=Path(directory); root.mkdir(parents=True,exist_ok=True)
    expected=json.loads(LOCK.read_text()); lock=root/'sources.json'
    previous=json.loads(lock.read_text()) if lock.exists() else {}; result={}
    for key, spec in expected.items():
        target=root/(key+'.json'); old=previous.get(key,{})
        revision=spec['revision'] or old.get('revision')
        if target.is_file() and checksum(target)==spec['sha256']:
            result[key]=dict(spec,revision=revision)
        else:
            if local_only: raise RuntimeError(f'Missing or wrong SHA256: {target}')
            from huggingface_hub import HfApi,hf_hub_download
            revision=revision or HfApi().dataset_info(spec['repo']).sha
            path=hf_hub_download(repo_id=spec['repo'],repo_type='dataset',filename=spec['filename'],revision=revision)
            if checksum(path)!=spec['sha256']:
                raise RuntimeError(f'Raw source differs from locked content: {key}; use the verified original JSON and its sources.json')
            temp=target.with_suffix('.json.part');shutil.copy2(path,temp);temp.replace(target)
            result[key]=dict(spec,revision=revision)
        previous[key]=result[key];lock.write_text(json.dumps(previous,indent=2)+'\n')
    return result


def records(path):
    path=Path(path)
    if path.suffix=='.parquet':yield from iter_rl(path);return
    if path.suffix=='.jsonl':
        with path.open() as f:
            for line in f:
                if line.strip():yield json.loads(line)
        return
    value=json.loads(path.read_text(encoding='utf-8'))
    if isinstance(value,dict):value=value.get('data',value.get('train'))
    if not isinstance(value,list):raise ValueError(f'Expected array of records: {path}')
    yield from value


def grouped_splits(rows, origins, seed, test_fraction=.1, sft_fraction=.2):
    """Connected components: shared source conversation OR any normalized user message."""
    if not 0<test_fraction<1 or not 0<sft_fraction<1 or test_fraction+sft_fraction>=1:
        raise ValueError('invalid split fractions')
    parent=list(range(len(rows)))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    seen={}
    for i,(row,origin) in enumerate(zip(rows,origins)):
        for key in [('origin',origin[:2]),*[('query',q) for q in user_keys(row['prompt'])]]:
            if key in seen:union(i,seen[key])
            else:seen[key]=i
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):groups[find(i)].append(i)
    # SFT format has a system message and a single user message; use only single-turn rows.
    def sft_eligible(r):
        if len(r['prompt'])!=2 or r['prompt'][-1]['role']!='user':return False
        try:parse('<think>I will select the appropriate tool and arguments.</think>\n<tool_call>'+r['answer']+'</tool_call>',strict=True);return True
        except (ValueError,TypeError,KeyError):return False
    eligible=[g for g,ix in groups.items() if any(sft_eligible(rows[i]) for i in ix)]
    if len(groups)<3 or not eligible:raise ValueError('Not enough independent groups for SFT/train/test')
    rng=random.Random(seed);rng.shuffle(eligible)
    target=max(1,round(len(rows)*sft_fraction));supervised=set();n=0
    for g in eligible:
        if len(supervised)>=len(groups)-2:break
        supervised.add(g);n+=sum(sft_eligible(rows[i]) for i in groups[g])
        if n>=target:break
    rest=[g for g in groups if g not in supervised];rng.shuffle(rest)
    held=set();n=0;target=max(1,round(len(rows)*test_fraction))
    for g in rest[:-1]:
        held.add(g);n+=len(groups[g])
        if n>=target:break
    out={'train':[],'test':[],'sft':[]}; omitted=0
    for g,ix in groups.items():
        split='sft' if g in supervised else 'test' if g in held else 'train'
        group_id=digest(sorted({origins[i][:2] for i in ix}))
        for i in ix:
            row=rows[i]; row=dict(row,extra_info=dict(row['extra_info'],group_id=group_id))
            if split=='sft' and not sft_eligible(row):
                omitted+=1;continue
            out[split].append(row)
    if not all(out.values()):raise ValueError('empty split after grouping')
    return out,dict(groups=len(groups),largest_group=max(map(len,groups.values())),sft_ineligible_omitted=omitted)


def write_parquet(path,rows,split):
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema=arrow_schema()
    with pq.ParquetWriter(path,schema,compression='snappy') as writer:
        for i in range(0,len(rows),512):
            batch=[dict(r,extra_info=dict(r['extra_info'],index=j,split=split)) for j,r in enumerate(rows[i:i+512],i)]
            writer.write_table(pa.Table.from_pylist(batch,schema=schema))


def make_sft(row):
    return dict(system_prompt=row['prompt'][0]['content'],query=row['prompt'][1]['content'],instruction='',
                output='<think>I will select the appropriate tool and arguments.</think>\n<tool_call>'+row['answer']+'</tool_call>',group_id=row['extra_info']['group_id'])


def rebuild(toolace,xlam,out,seed=42,test_fraction=.1,sft_fraction=.2,include_nocall=False,strict_schema=False,source_lock=None):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(f'{out} already exists; choose a new output or move the old generated dataset')
    inputs={'toolace':Path(toolace),'xlam':Path(xlam)};rows=[];entries=[];report=collections.Counter();origin=[]
    with tempfile.TemporaryDirectory(dir=out.parent,prefix='.build-') as temp:
        stage=Path(temp)/'dataset';stage.mkdir()
        with (stage/'rejected.jsonl').open('w') as rejected,(stage/'warnings.jsonl').open('w') as warnings:
            for source,path in inputs.items():
                for index,r in enumerate(records(path)):
                    report[source+'_input']+=1
                    try:
                        converted=list(from_toolace(r,include_nocall)) if source=='toolace' else [from_xlam(r)]
                        if not converted:raise ValueError('no eligible tool call')
                        # Validate all rows before adding any from this conversation.
                        candidates=[to_rl_row(rec,'candidate',len(rows)+n) for n,rec in enumerate(converted)]
                        for n,row in enumerate(candidates):
                            entry=inspect(row,'train');entry['row']=len(rows)
                            rows.append(row);entries.append(entry);origin.append((source,index,n));report[source+'_converted']+=1
                            if entry['warnings']:warnings.write(json.dumps(dict(source=source,index=index,turn=n,warnings=entry['warnings']))+'\n')
                    except (ValueError,TypeError,KeyError,IndexError,AttributeError,OverflowError,RecursionError) as exc:
                        report[source+'_invalid']+=1;rejected.write(json.dumps(dict(source=source,index=index,reason=str(exc)))+'\n')
            for source in inputs:
                if not report[source+'_converted']:raise ValueError(f'{source}: zero converted rows; check input format')
            labels=collections.defaultdict(set)
            for e in entries:
                if e['key']:labels[e['key']].add(e['label'])
            conflicts={k for k,v in labels.items() if len(v)>1}
            kept,removed=decisions(entries,conflicts,set(),set(),strict_schema)
            for r in removed:
                source,index,n=origin[r['row']];rejected.write(json.dumps(dict(source=source,index=index,turn=n,**r))+'\n')
        # Group before dropping duplicates so duplicate rows cannot disconnect a dialogue.
        # Retain all valid rows as grouping links, but exclude rejected rows from outputs.
        # Exact prompt duplicates share all user keys, so include their origin links too.
        valid=[i for i,e in enumerate(entries) if e['key'] and not e['hard']]
        # Build component ids over all valid original candidates, including conflicting/duplicate rows.
        parent=list(range(len(rows)));seen={}
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        for i in valid:
            for k in [('o',origin[i][:2]),*[('q',q) for q in user_keys(rows[i]['prompt'])]]:
                if k in seen:
                    a,b=find(i),find(seen[k]);parent[max(a,b)]=min(a,b)
                else:seen[k]=i
        effective_origins=[('component',find(i),0) for i in kept]
        splits,group_stats=grouped_splits([rows[i] for i in kept],effective_origins,seed,test_fraction,sft_fraction)
        for name in ('train','test'):write_parquet(stage/(name+'.parquet'),splits[name],name)
        (stage/'sft.json').write_text(json.dumps([make_sft(r) for r in splits['sft']],ensure_ascii=False,indent=2)+'\n')
        meta=dict(raw_counts=dict(report),accepted=len(kept),rejected=len(removed),seed=seed,test_fraction=test_fraction,sft_fraction=sft_fraction,strict_schema=strict_schema,include_nocall=include_nocall,
                  **group_stats,**{k:len(v) for k,v in splits.items()},sft_policy='deterministic format supervision; no teacher traces',source_hashes={k:checksum(v) for k,v in inputs.items()},sources=source_lock or {},
                  rejection_reasons=dict(collections.Counter(reason for r in removed for reason in r['reasons'])),output_hashes={k:checksum(stage/k) for k in ('train.parquet','test.parquet','sft.json')})
        (stage/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');stage.rename(out)
    print(json.dumps(meta,indent=2));return meta


def main():
    p=argparse.ArgumentParser();p.add_argument('--originals',default='data/originals');p.add_argument('--local',action='store_true');p.add_argument('--out',default='data/processed');p.add_argument('--test-fraction',type=float,default=.1);p.add_argument('--sft-fraction',type=float,default=.2);p.add_argument('--seed',type=int,default=42);p.add_argument('--include-nocall',action='store_true');p.add_argument('--strict-schema',action='store_true');a=p.parse_args()
    root=Path(a.originals);lock=fetch_originals(root,a.local);rebuild(root/'toolace.json',root/'xlam.json',a.out,a.seed,a.test_fraction,a.sft_fraction,a.include_nocall,a.strict_schema,lock)

if __name__=='__main__':main()
