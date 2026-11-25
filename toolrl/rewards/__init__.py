"""Local binary baseline, structure-aware reward and common evaluation."""
import json
import math
import re
from collections import Counter


def _no_duplicates(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f'duplicate key: {k}')
        out[k] = v
    return out


def _bad_constant(v):
    raise ValueError(f'non-finite JSON: {v}')


def loads(text):
    return json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_bad_constant)


def calls(value):
    for _ in range(2):
        if isinstance(value, str):
            value = loads(value)
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise ValueError('expected list or object')
    for c in value:
        if not isinstance(c, dict) or set(c) != {'name', 'arguments'} or not isinstance(c['name'], str) or not c['name'] or not isinstance(c['arguments'], dict):
            raise ValueError('invalid call structure')
    return value


def canon(v):
    # Numeric equivalence without conflating bool with int or changing strings.
    if v is None: return ('null',)
    if isinstance(v, bool): return ('bool', v)
    if isinstance(v, (int, float)):
        if isinstance(v, float) and not math.isfinite(v): raise ValueError('nonfinite number')
        return ('number', v)
    if isinstance(v, str): return ('string', v)
    if isinstance(v, list): return ('array', tuple(map(canon, v)))
    if isinstance(v, dict): return ('object', tuple((k, canon(x)) for k, x in sorted(v.items())))
    raise ValueError('unsupported type')


def match(a, b):
    # Single-step / parallel calls: order invariant, multiplicity preserved.
    return Counter(map(canon, a)) == Counter(map(canon, b))


def assistant_text(text):
    if '<|im_start|>assistant' in text:
        text = text.rsplit('<|im_start|>assistant', 1)[1]
    return re.sub(r'(?:\s*(?:<\|im_end\|>|<\|endoftext\|>))+\s*$', '', text).strip()


def parse(text, strict=False):
    text = assistant_text(text)
    if strict:
        m = re.fullmatch(r'<think>((?:(?!</?think>|</?tool_call>).)*)</think>\s*<tool_call>((?:(?!</?think>|</?tool_call>).)*)</tool_call>', text, flags=re.S)
        if not m: raise ValueError('expected one ordered think/call pair')
        return calls(loads(m.group(2)))
    # Independent, common evaluator: allows native Qwen multiple call blocks,
    # or bare JSON; no reasoning tag requirement. Never silently drops a block.
    blocks = re.findall(r'<tool_call>(.*?)</tool_call>', text, flags=re.S)
    if blocks:
        if text.count('<tool_call>') != len(blocks) or text.count('</tool_call>') != len(blocks):
            raise ValueError('unbalanced call tags')
        result = []
        for b in blocks: result.extend(calls(loads(b)))
        return result
    if '<tool_call' in text or '</tool_call' in text: raise ValueError('bad tags')
    return calls(loads(text))


def structured_score(solution_str, ground_truth):
    try: return float(match(parse(solution_str, strict=True), calls(ground_truth)))
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError): return 0.0


from .baseline import baseline_score


def evaluate(text, truth):
    target = calls(truth)
    try:
        pred = parse(text)
        names = Counter(c['name'] for c in pred) == Counter(c['name'] for c in target)
        exact = match(pred, target)
        valid = True
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
        exact = names = valid = False
    try:
        parse(text, strict=True)
        strict = True
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError): strict = False
    return dict(exact_match=int(exact), tool_match=int(names), valid_call=int(valid), strict_format=int(strict))
