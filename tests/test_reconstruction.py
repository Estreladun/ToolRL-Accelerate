import json
import pytest
from toolrl.data.converter import (TEMPLATE,MARKER,parse_function_calls,extract_system,
                                   from_toolace,from_xlam,to_rl_row,prompt_from_raw)
from toolrl.data.quality import inspect,scan,decisions,prompt_tools
from toolrl.data.reconstruct import grouped_splits

TOOLS=[{'name':'get place','parameters':{'type':'object','properties':{'city':{'type':'string'},'n':{'type':'int'}},'required':['city']}}]

def row(q='hello',answer=None):
    return to_rl_row(dict(raw_system=TEMPLATE['default_system'],tools=TOOLS,conversations=[{'from':'human','value':q}],answer=answer or [{'name':'get place','arguments':{'city':'X'}}]),'train',0)

def test_prompt_reconstruction_and_quality():
    r=row(); assert set(r)=={'tools','conversations','answer','raw_system','data_source','prompt','ability','reward_model','extra_info'}
    assert prompt_tools(r['prompt'][0]['content'])==TOOLS
    d=inspect(r,'train');assert d['rebuilt'] and not d['hard'] and not d['warnings']

def test_nested_and_escaped_calls():
    a=parse_function_calls(r'''[get place(city="A, (B)", n=2, extra={"x":[1,2],"quoted":"a\"b"})]''')
    assert a[0]['arguments']=={'city':'A, (B)','n':2,'extra':{'x':[1,2],'quoted':'a"b'}}

@pytest.mark.parametrize('bad',['[f(x=1,x=2)]','[f(x=__import__("os"))]','[f(x={"a":1,"a":2})]','[f(x=[1,2)]'])
def test_reject_bad_calls(bad):
    with pytest.raises(ValueError):parse_function_calls(bad)

def test_toolace_roles_multiturn():
    system=TEMPLATE['default_system']+json.dumps(TOOLS)+'\nShould you decide to return the function call(s).'
    source={'system':system,'conversations':[{'from':'user','value':'first'},{'from':'assistant','value':'[get place(city="A")]'},{'from':'tool','value':'result'},{'from':'user','value':'second'},{'from':'assistant','value':'[get place(city="B")]'}]}
    out=list(from_toolace(source));assert len(out)==2
    p=to_rl_row(out[1],'train',1)['prompt'];assert [m['role'] for m in p]==['system','user','assistant','tool','user']
    assert json.loads(p[2]['content'])==out[1]['conversations'][1]['value']

def test_xlam_shapes():
    r=from_xlam({'query':'hello','tools':json.dumps(TOOLS),'answers':json.dumps([{'name':'get place','arguments':{'city':'X'}}])}); assert r['tools']==TOOLS
    with pytest.raises(ValueError):from_xlam({'query':'hello','tools':TOOLS,'answers':[{'name':'missing','arguments':{}}]})

def test_warnings_not_auto_reject():
    r=row(answer=[{'name':'get place','arguments':{'n':'bad'}}]);d=inspect(r,'train');assert set(d['warnings'])=={'missing_required_parameter','parameter_type_mismatch'} and not d['hard']
    d['row']=0;assert decisions([d],set(),set(),set())[0]==[0];assert decisions([d],set(),set(),set(),True)[0]==[]

def test_conflict_duplicate_leakage():
    a=row('a');b=row('b');c=row('c');d=row('c',answer=[{'name':'get place','arguments':{'city':'Y'}}]);te=row('b')
    e,stats,_,_,conflicts=scan([a,a,b,c,d],'train');_,_,tk,tq,_=scan([te],'test')
    keep,reject=decisions(e,conflicts,tk,tq);assert keep==[0];assert len(reject)==4;assert stats['conflicting_prompt_keys']==1

def test_group_split_no_query_overlap():
    from toolrl.data.preprocess import user_keys
    data=[row(str(i)) for i in range(8)]+[row('0')]
    origins=[('source',i,0) for i in range(len(data))]
    splits,_=grouped_splits(data,origins,42,.25,.25)
    assert sum(map(len,splits.values()))==len(data)
    keys=[set().union(*(user_keys(r['prompt']) for r in v)) for v in splits.values()]
    assert all(not a&b for i,a in enumerate(keys) for b in keys[i+1:])

def test_sft_flags_and_tool_block():
    sys=prompt_from_raw(TEMPLATE['default_system'],TOOLS,[])[0]['content']
    r={'system_prompt':sys,'query':'hello','instruction':'','output':'<think>x</think><tool_call>[{"name":"get place","arguments":{"city":"A"}}]</tool_call>'}
    assert not inspect(r,'sft')['hard'];r['output']='bad';assert inspect(r,'sft')['hard']

def test_rebuild_from_raw_to_parquet(tmp_path):
    pytest.importorskip('pyarrow')
    from toolrl.data.reconstruct import rebuild
    from toolrl.data.quality import iter_rl
    from toolrl.data.preprocess import user_keys,rl_rows,sft_rows
    a=tmp_path/'toolace.json'
    a.write_text(json.dumps([{'system':TEMPLATE['default_system']+json.dumps(TOOLS)+'. \nShould you decide to return the function call(s).','conversations':[{'from':'user','value':'query1'},{'from':'assistant','value':'[get place(city="X")]'}]}]))
    b=tmp_path/'xlam.json'
    b.write_text(json.dumps([{'query':f'query{i}','tools':TOOLS,'answers':[{'name':'get place','arguments':{'city':'X'}}]} for i in range(2,22)]))
    out=tmp_path/'out';res=rebuild(a,b,out)
    assert res['raw_counts']['toolace_converted']==1
    assert res['train']+res['test']+res['sft']==21
    groups=[{r['group_id'] for r in fn(out/file)} for fn,file in [(rl_rows,'train.parquet'),(rl_rows,'test.parquet'),(sft_rows,'sft.json')]]
    assert all(not x&y for i,x in enumerate(groups) for y in groups[i+1:])
    from toolrl.data.validate import verify
    assert verify(out)['train']['rows']==res['train']


def test_actual_punctuation_and_role_aliases():
    sys=TEMPLATE['default_system']+json.dumps(TOOLS)+'. \nShould you decide to return the function call(s). \nPut it in the format of [f(x=1)]'
    source={'system':sys,'conversations':[{'from':'user','value':'first'},{'from':'assistant','value':'[get place(city="A")]'},{'from':'tool','value':'result'},{'from':'assistant','value':'plain answer'},{'from':'user','value':'second'},{'from':'assistant','value':'[get place(city="B")]'}]}
    assert extract_system(sys)[1]==TOOLS
    rows=list(from_toolace(source));assert len(rows)==2
    assert len(rows[1]['conversations'])==5


def test_legacy_marker_and_trailing_comma():
    assert extract_system('Functions:\n['+json.dumps(TOOLS[0])+',].\nInstructions')[1]==TOOLS


def test_group_dialogue_and_query_transitive():
    data=[row(str(i)) for i in range(10)]
    origins=[('source',i,0) for i in range(len(data))]
    origins[1]=('source',0,1)
    data[2]=row('1')
    splits,_=grouped_splits(data,origins,42,.2,.2)
    memberships={r['prompt'][-1]['content']:k for k,rs in splits.items() for r in rs}
    assert memberships['0']==memberships['1']
