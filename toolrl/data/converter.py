"""Reconstruct released-format RL rows from raw ToolACE/xLAM.
Data format and prompt template reconstructed from the released records.
Public parsing/splitting approach is described in README; no dynamic evaluation.
"""
import ast
import json
import re
from pathlib import Path
from ..rewards import calls, loads, canon

TEMPLATE_FILE = Path(__file__).resolve().parents[2] / 'configs/prompt_template.json'
TEMPLATE = json.loads(TEMPLATE_FILE.read_text())
MARKER = 'Here is a list of functions in JSON format that you can invoke:\n'
ROLES = {'human':'user','gpt':'assistant','function_call':'assistant','observation':'tool',
         'user':'user','assistant':'assistant','tool':'tool'}


def json_value(x):
    return loads(x) if isinstance(x, str) else x


def normalize_tools(value):
    value = json_value(value)
    if not isinstance(value, list) or not value:
        raise ValueError('empty or invalid tools')
    result = []
    for tool in value:
        if not isinstance(tool, dict): raise ValueError('invalid tool')
        if tool.get('type') == 'function' and isinstance(tool.get('function'),dict):
            tool = tool['function']
        if not isinstance(tool.get('name'),str) or not tool['name'].strip():
            raise ValueError('missing tool name')
        if 'parameters' in tool and not isinstance(tool['parameters'],dict):
            raise ValueError('invalid parameter definition')
        canon(tool)
        result.append(tool)
    return result


def split_balanced(text, separator=','):
    """Split only outside strings and nested delimiters; escaped quotes supported."""
    stack=[]; quote=None; escape=False; start=0; out=[]
    pairs={')':'(',']':'[','}':'{'}
    for i,ch in enumerate(text):
        if quote:
            if escape: escape=False
            elif ch=='\\': escape=True
            elif ch==quote: quote=None
        elif ch in ('"', "'"): quote=ch
        elif ch in '([{': stack.append(ch)
        elif ch in ')]}':
            if not stack or stack.pop()!=pairs[ch]: raise ValueError('unbalanced delimiters')
        elif ch==separator and not stack:
            out.append(text[start:i].strip()); start=i+1
    if stack or quote: raise ValueError('unbalanced literal')
    out.append(text[start:].strip())
    return out


def literal(value):
    try: v=loads(value)
    except (ValueError,TypeError):
        try:
            node=ast.parse(value,mode='eval')
            # Reject duplicate dictionary keys rather than letting literal_eval overwrite them.
            for n in ast.walk(node):
                if isinstance(n,ast.Dict):
                    keys=[ast.literal_eval(k) for k in n.keys]
                    if len(set(keys))!=len(keys): raise ValueError('duplicate literal key')
            v=ast.literal_eval(node)
        except (SyntaxError,ValueError,TypeError,MemoryError,RecursionError) as exc:
            # ToolACE occasionally uses unquoted bare tokens. Do not accept executable expressions.
            if re.fullmatch(r'[A-Za-z0-9_.:/-]+',value): v=value
            else: raise ValueError('unsupported argument literal') from exc
    # Tuples become JSON arrays. Reject sets/bytes/non-string dict keys instead of guessing.
    def safe(x):
        if isinstance(x,tuple): return [safe(y) for y in x]
        if isinstance(x,list): return [safe(y) for y in x]
        if isinstance(x,dict):
            if not all(isinstance(k,str) for k in x): raise ValueError('non-string key')
            return {k:safe(y) for k,y in x.items()}
        if x is None or isinstance(x,(str,int,float,bool)): return x
        raise ValueError('non-JSON literal')
    v=safe(v); canon(v); return v


def parse_function_calls(text):
    try:
        result=calls(text); canon(result); return result
    except (ValueError,TypeError,KeyError): pass
    if not isinstance(text,str): raise ValueError('invalid call text')
    text=text.strip()
    if not (text.startswith('[') and text.endswith(']')): raise ValueError('expected bracketed calls')
    result=[]
    for part in split_balanced(text[1:-1]):
        if not part: continue
        m=re.fullmatch(r'([\w .?\|/\-]+)\((.*)\)',part,flags=re.S)
        if not m: raise ValueError('invalid function call')
        args={}
        for item in split_balanced(m[2]):
            if not item: continue
            key,sep,val=item.partition('=');key=key.strip()
            if not sep or not key or key in args: raise ValueError('invalid or duplicate parameter')
            args[key]=literal(val.strip())
        result.append({'name':m[1].strip(),'arguments':args})
    return calls(result)


def extract_system(system):
    """Parse a JSON tool array, allowing trailing prose and historical markers."""
    if not isinstance(system,str): raise ValueError('invalid system prompt')
    marker=re.search(r'Here is a list of functions in JSON format that you can invoke:[ \t]*\r?\n?',system,re.I)
    start=marker.end() if marker else 0
    prefix=system[:start] if marker else TEMPLATE['default_system']
    # The current ToolACE release has a literal period after the JSON array.
    # raw_decode consumes exactly one JSON value, leaving trailing instructions alone.
    from ..rewards import _no_duplicates, _bad_constant
    dec=json.JSONDecoder(object_pairs_hook=_no_duplicates,parse_constant=_bad_constant)
    body=system[start:]
    for candidate in re.finditer(r'\[\s*\{',body):
        text=body[candidate.start():]
        try:
            value,_=dec.raw_decode(text)
            return prefix,normalize_tools(value)
        except (ValueError,TypeError,RecursionError):
            # Legacy lists may have a trailing comma. Only remove an actual comma
            # preceding a closing bracket, outside strings, by decoding each object.
            try:
                pos=1; objects=[]
                while True:
                    while pos<len(text) and text[pos].isspace():pos+=1
                    if pos<len(text) and text[pos]==']':break
                    obj,pos=dec.raw_decode(text,pos);objects.append(obj)
                    while pos<len(text) and text[pos].isspace():pos+=1
                    if pos<len(text) and text[pos]==']':break
                    if pos>=len(text) or text[pos]!=',':raise ValueError('invalid tool list')
                    pos+=1
                if pos<len(text) and text[pos]==']':return prefix,normalize_tools(objects)
            except (ValueError,TypeError,RecursionError):pass
            # Do not interpret nested arrays inside a malformed outer tool list.
            break
    raise ValueError('missing or invalid JSON tool list')


def validate_target(tools,answer):
    answer=calls(answer); canon(answer)
    names={t['name'] for t in tools}
    if any(c['name'] not in names for c in answer): raise ValueError('unknown target tool')
    return answer


def from_xlam(record):
    tools=normalize_tools(record['tools']); answer=validate_target(tools,json_value(record['answers']))
    query=record['query']
    if not isinstance(query,str) or not query.strip(): raise ValueError('empty query')
    return dict(raw_system=TEMPLATE['default_system'],tools=tools,conversations=[{'from':'human','value':query}],answer=answer)


def from_toolace(record,include_nocall=False):
    system,tools=extract_system(record['system']); history=[]
    for turn in record['conversations']:
        role,value=turn['from'],turn['value']
        role={'user':'human','assistant':'gpt','tool':'observation','function':'observation'}.get(role,role)
        if not isinstance(value,str) or role not in ROLES: raise ValueError('invalid conversation turn')
        call=None
        if role=='function_call' or (role=='gpt' and value.strip().startswith('[') and value.strip().endswith(']')):
            try: call=parse_function_calls(value)
            except (ValueError,TypeError,KeyError):
                if role=='function_call' or re.match(r'^\s*\[\s*[\w .?\|/\-]+\(',value): raise
            if call is not None:
                call=validate_target(tools,call)
                if history: yield dict(raw_system=system,tools=tools,conversations=list(history),answer=call)
                role='function_call'; value=json.dumps(call)
        elif role=='gpt' and include_nocall and history:
            yield dict(raw_system=system,tools=tools,conversations=list(history),answer=[])
        history.append({'from':role,'value':value})


def prompt_from_raw(raw_system,tools,conversations):
    system=raw_system+TEMPLATE['tool_prefix']+'\n'.join(json.dumps({'type':'function','function':t}) for t in tools)+TEMPLATE['tool_suffix']
    prompt=[{'role':'system','content':system}]
    for turn in conversations:
        role=turn['from']; value=turn['value']
        # Released records double-encode historical function calls as JSON strings.
        if role=='function_call': value=json.dumps(value)
        prompt.append({'role':ROLES[role],'content':value})
    return prompt


def to_rl_row(record,split,index):
    tools=normalize_tools(record['tools']); answer=validate_target(tools,record['answer'])
    truth=json.dumps(answer)
    return dict(tools=json.dumps(tools),conversations=record['conversations'],
                answer=truth,raw_system=record['raw_system'],data_source='toolcall',
                prompt=prompt_from_raw(record['raw_system'],tools,record['conversations']),ability='toolcall',
                reward_model={'ground_truth':truth,'style':'rule'},extra_info={'index':index,'split':split})


def arrow_schema():
    import pyarrow as pa
    string=pa.string(); turn=lambda a,b:pa.list_(pa.struct([(a,string),(b,string)]))
    return pa.schema([('tools',string),('conversations',turn('from','value')),('answer',string),('raw_system',string),('data_source',string),('prompt',turn('content','role')),('ability',string),('reward_model',pa.struct([('ground_truth',string),('style',string)])),('extra_info',pa.struct([('index',pa.int64()),('split',string),('group_id',string)]))])
