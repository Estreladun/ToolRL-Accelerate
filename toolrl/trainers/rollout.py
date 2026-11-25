"""Batched generation with completion lengths measured before padding."""

def trim_completion(seq, prompt_len, eos):
    eos = set(eos if isinstance(eos, (tuple,list)) else [eos])
    ids = seq[0,prompt_len:].tolist()
    for i, token in enumerate(ids):
        if token in eos:
            return seq[:,:prompt_len+i+1]
    return seq


def generate_group(model,tok,ids,a,count,sample=True):
    import torch
    kwargs=dict(max_new_tokens=a.max_response,do_sample=sample,use_cache=True,
                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    eos=model.generation_config.eos_token_id
    if eos is None: eos=tok.eos_token_id
    kwargs['eos_token_id']=eos
    if sample: kwargs.update(temperature=a.temperature,top_p=1.0,top_k=0)
    else: kwargs.update(temperature=1.0,top_p=1.0,top_k=None)
    model.eval()
    batch=max(1,min(getattr(a,'rollout_batch',1),count)) if sample else 1
    for start in range(0,count,batch):
        with torch.no_grad():
            outputs=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
                                   num_return_sequences=min(batch,count-start),**kwargs)
        for seq in outputs.split(1):
            seq=trim_completion(seq,ids.shape[1],eos)
            suffix=seq[0,ids.shape[1]:]
            yield seq,tok.decode(suffix,skip_special_tokens=True),suffix.numel()


def synchronize(device):
    import torch
    if device.type=='cuda': torch.cuda.synchronize(device)
    elif device.type=='mps': torch.mps.synchronize()
