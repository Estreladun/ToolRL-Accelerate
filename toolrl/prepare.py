"""Build a shared, tokenized experiment data cache."""
import argparse
import json
from pathlib import Path
from .data.preprocess import digest,rl_rows,sft_rows,build_splits
from .resources import checksum


def main():
    from transformers import AutoTokenizer
    p=argparse.ArgumentParser()
    for name in ('train-file','test-file','sft-file','out','model'): p.add_argument('--'+name,required=True)
    for name in ('train','dpo','pool','val','test','max-prompt','max-response','sft-response','seed'): p.add_argument('--'+name,type=int,required=True)
    a=p.parse_args()
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    splits=build_splits(rl_rows(a.train_file),rl_rows(a.test_file),sft_rows(a.sft_file),tok,sft_n=a.train,dpo_n=a.dpo,pool_n=a.pool,val_n=a.val,test_n=a.test,max_prompt=a.max_prompt,max_response=a.max_response,sft_response=a.sft_response,seed=a.seed)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    manifest=dict(seed=a.seed,source_hashes={k:checksum(getattr(a,k+'_file')) for k in ('train','test','sft')},max_prompt=a.max_prompt,max_response=a.max_response,sft_response=a.sft_response)
    for name,records in splits.items():
        (out/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
        manifest[name]=len(records); manifest[name+'_ids']=digest([r['id'] for r in records])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)); print(manifest)

if __name__=='__main__':main()
