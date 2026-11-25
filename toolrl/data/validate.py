"""Verify generated files, output counts, and cross-split group isolation."""
import argparse,json
from pathlib import Path
from .preprocess import rl_rows,sft_rows,user_keys
from ..resources import checksum


def verify(directory):
    root=Path(directory);m=json.loads((root/'manifest.json').read_text());sets={};stats={}
    for split,filename,reader in [('train','train.parquet',rl_rows),('test','test.parquet',rl_rows),('sft','sft.json',sft_rows)]:
        path=root/filename
        if checksum(path)!=m['output_hashes'][filename]:raise ValueError(f'Checksum mismatch: {path}')
        groups=set();queries=set();count=0
        for row in reader(path):
            if not row['group_id']:raise ValueError('Missing group_id')
            count+=1;groups.add(row['group_id']);queries.update(user_keys(row['prompt']))
        if count!=m[split] or count==0:raise ValueError(f'{split}: readable rows {count}, expected {m[split]}')
        sets[split]=(groups,queries);stats[split]=dict(rows=count,groups=len(groups),user_keys=len(queries))
    for i,a in enumerate(sets):
        for b in list(sets)[i+1:]:
            if any(x&y for x,y in zip(sets[a],sets[b])):raise ValueError(f'Leakage: {a}/{b}')
    for source in ('toolace','xlam'):
        if not m['raw_counts'].get(source+'_converted'):raise ValueError(f'{source}: no converted rows')
    print(json.dumps(dict(status='PASS',splits=stats,group_overlap=0,normalized_user_overlap=0),indent=2));return stats

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data/processed');verify(p.parse_args().data_dir)
