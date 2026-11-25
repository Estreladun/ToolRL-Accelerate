import json
from toolrl.rewards import structured_score, evaluate, calls, match
T=json.dumps([{'name':'weather','arguments':{'temp':20,'city':'Nanjing'}}])
GOOD='<think>need weather</think><tool_call>'+T+'</tool_call>'

def test_good(): assert structured_score(GOOD,T)==1

def test_float_equivalent(): assert structured_score(GOOD.replace('20','20.0'),T)==1

def test_extra_call(): assert structured_score(GOOD+GOOD,T)==0

def test_wrong_order(): assert structured_score('<tool_call>'+T+'</tool_call><think>x</think>',T)==0

def test_no_reason(): assert structured_score('<tool_call>'+T+'</tool_call>',T)==0

def test_wrong_value(): assert structured_score(GOOD.replace('Nanjing','Beijing'),T)==0

def test_duplicate_key(): assert structured_score(GOOD.replace('"temp": 20','"temp": 20, "temp": 20'),T)==0

def test_no_boolean_number_confusion(): assert not match(calls([{'name':'f','arguments':{'x':True}}]),calls([{'name':'f','arguments':{'x':1}}]))

def test_unordered_and_multiplicity():
    a=calls(T); b=[{'name':'f','arguments':{}}]; assert match(a+b,b+a); assert not match(a+b,a+b+b)

def test_native_eval(): assert evaluate('<tool_call>'+T+'</tool_call>',T)==dict(exact_match=1,tool_match=1,valid_call=1,strict_format=0)

def test_empty(): assert structured_score('<think>no tool</think><tool_call>[]</tool_call>','[]')==1

def test_nonfinite(): assert structured_score(GOOD.replace('20','NaN'),T)==0

def test_suffix(): assert structured_score('<|im_start|>assistant\n'+GOOD+'<|im_end|>',T)==1

def test_eval_multiple():
    assert evaluate(GOOD+GOOD,T)['exact_match']==0

from toolrl.rewards import baseline_score

def test_binary_baseline_correct(): assert baseline_score(GOOD,T)==1

def test_baseline_vs_sar_duplicates():
    bad=GOOD.replace('"temp": 20','"temp": 0, "temp": 20')
    assert baseline_score(bad,T)==1 and structured_score(bad,T)==0

def test_baseline_vs_sar_last_block():
    bad=GOOD+GOOD
    assert baseline_score(bad,T)==1 and structured_score(bad,T)==0

def test_baseline_vs_sar_numeric():
    v=GOOD.replace('20','20.0')
    assert baseline_score(v,T)==0 and structured_score(v,T)==1
