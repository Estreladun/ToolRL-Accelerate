"""Conservative validation and deduplication for generated data.
Never execute tools or interpret user queries as code.
"""
import collections
import json
from pathlib import Path
from .converter import normalize_tools, validate_target, prompt_from_raw
from .preprocess import digest, query_key
from ..rewards import calls, canon, loads, parse, match


def iter_rl(path):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        yield from batch.to_pylist()


def iter_sft(path):
    yield from loads(Path(path).read_text(encoding='utf-8'))


def prompt_tools(system):
    import re
    block=re.search(r'^\s*<tools>[ \t]*\n(.*?)^\s*</tools>',system,re.S|re.M)
    if not block: raise ValueError('missing tools block')
    return normalize_tools([loads(line) for line in block[1].splitlines() if line.strip()])


def value_type_ok(value, name):
    # Legacy Python type hints in this corpus are not valid JSON Schema.
    # Only check recognized shallow types; unknown or union forms stay unclassified.
    if not isinstance(name,str): return None
    t=name.split(', optional',1)[0].strip().lower()
    if t.startswith(('list[','tuple[')): t='list'
    if t.startswith('dict['): t='dict'
    checks={'str':lambda x:isinstance(x,str),'string':lambda x:isinstance(x,str),
      'int':lambda x:type(x) is int,'integer':lambda x:type(x) is int,
      'float':lambda x:type(x) in (float,int),'number':lambda x:type(x) in (float,int),
      'bool':lambda x:type(x) is bool,'boolean':lambda x:type(x) is bool,
      'list':lambda x:isinstance(x,list),'array':lambda x:isinstance(x,list),
      'dict':lambda x:isinstance(x,dict),'object':lambda x:isinstance(x,dict),
      'null':lambda x:x is None}
    return checks[t](value) if t in checks else None


def schema_warnings(tools,truth):
    warnings=set(); by_name=collections.defaultdict(list)
    for t in tools:by_name[t['name']].append(t)
    if any(len(ts)>1 for ts in by_name.values()): warnings.add('duplicate_tool_name')
    for call in truth:
        ts=by_name[call['name']]
        # Ambiguous definitions must be reviewed, not silently selected.
        if len(ts)>1 and any(t!=ts[0] for t in ts[1:]):warnings.add('ambiguous_tool_definition');continue
        p=ts[0].get('parameters',{});args=call['arguments']
        if not isinstance(p,dict): warnings.add('unsupported_schema');continue
        schema='properties' in p
        props=p.get('properties',{}) if schema else p
        if not isinstance(props,dict):warnings.add('unsupported_schema');continue
        required=p.get('required',[]) if schema else ts[0].get('required',[])
        if isinstance(required,list) and any(k not in args for k in required):warnings.add('missing_required_parameter')
        if any(k not in props for k in args):warnings.add('unknown_parameter')
        for k,v in args.items():
            spec=props.get(k)
            if not isinstance(spec,dict):continue
            if value_type_ok(v,spec.get('type')) is False:warnings.add('parameter_type_mismatch')
            if isinstance(spec.get('enum'),list) and not any(canon(v)==canon(x) for x in spec['enum']):warnings.add('enum_mismatch')
    return warnings


def inspect(record,kind):
    hard=set(); warnings=set(); key=query=None; label=None; rebuilt=None
    try:
        if kind=='sft':
            prompt=[{'role':'system','content':record['system_prompt']},{'role':'user','content':'\n'.join(x for x in [record.get('instruction',''),record['query']] if x)}]
            text=record['output'].strip(); truth=parse(text,strict=True);canon(truth)
            tools=prompt_tools(prompt[0]['content'])
            if '<think></think>' in text or '<think>\n\n</think>' in text:warnings.add('empty_reasoning')
        else:
            prompt=record['prompt'];tools=normalize_tools(record['tools']);truth=calls(record['answer']);canon(truth)
            gt=calls(record['reward_model']['ground_truth']);canon(gt)
            if not match(truth,gt):hard.add('ground_truth_mismatch')
            if canon(prompt_tools(prompt[0]['content']))!=canon(tools):hard.add('prompt_tools_mismatch')
            rebuilt=prompt_from_raw(record['raw_system'],tools,record['conversations'])==prompt
            if not rebuilt:warnings.add('prompt_rebuild_difference')
        validate_target(tools,truth)
        key=digest(prompt);query=query_key(prompt);label=frozenset(collections.Counter(map(canon,truth)).items())
        warnings.update(schema_warnings(tools,truth))
    except (ValueError,TypeError,KeyError,IndexError,AttributeError,OverflowError,RecursionError) as e:
        hard.add('invalid_record');warnings.add('parse:'+str(e)[:120])
    return dict(key=key,query=query,label=label,hard=sorted(hard),warnings=sorted(warnings),rebuilt=rebuilt)


def scan(records,kind):
    entries=[];seen={};labels=collections.defaultdict(set);counts=collections.Counter();warning_counts=collections.Counter();queries=set();dups=0;query_dups=0;rebuild=0
    for i,r in enumerate(records):
        x=inspect(r,kind);x['row']=i;entries.append(x);counts.update(x['hard']);warning_counts.update(x['warnings']);rebuild+=int(x['rebuilt'] is True)
        k=x['key'];q=x['query']
        if k:
            if k in seen:dups+=1
            else:seen[k]=i
            labels[k].add(x['label'])
        if q:
            query_dups+=int(q in queries);queries.add(q)
    conflicts={k for k,l in labels.items() if len(l)>1}
    return entries,dict(rows=len(entries),hard_reasons=dict(counts),warning_reasons=dict(warning_counts),exact_duplicate_rows=dups,query_duplicate_rows=query_dups,conflicting_prompt_keys=len(conflicts),prompt_rebuilt_exact=rebuild),set(seen),queries,conflicts


def decisions(entries,conflicts,holdout_keys,holdout_queries,strict_schema=False):
    seen=set();accepted=[];rejected=[]
    strict={'missing_required_parameter','unknown_parameter','parameter_type_mismatch','enum_mismatch','ambiguous_tool_definition'}
    for x in entries:
        reason=set(x['hard']);k=x['key'];q=x['query']
        if k in conflicts:reason.add('conflicting_labels')
        if k and k in holdout_keys:reason.add('test_prompt_overlap')
        if q and q in holdout_queries:reason.add('test_query_overlap')
        if k and k in seen:reason.add('exact_duplicate')
        if strict_schema:reason.update(strict.intersection(x['warnings']))
        if reason:rejected.append(dict(row=x['row'],reasons=sorted(reason),warnings=x['warnings'],prompt_id=k));continue
        if k:seen.add(k)
        accepted.append(x['row'])
    return accepted,rejected

