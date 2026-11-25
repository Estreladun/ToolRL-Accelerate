"""Run on Linux: torchrun --standalone --nproc_per_node=2 scripts/check_dual_gpu.py"""
import os
from datetime import timedelta
import torch
import torch.distributed as dist

rank=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(rank)
assert torch.cuda.device_count()>=2, 'Need two visible GPUs'
props=torch.cuda.get_device_properties(rank)
print(rank,props.name,props.major,props.minor,round(props.total_memory/2**30,2),'GiB', 'native BF16',props.major>=8,flush=True)
dist.init_process_group('nccl',timeout=timedelta(seconds=60))
try:
    x=torch.full((1024,1024),rank+1.,device='cuda'); dist.all_reduce(x); assert torch.all(x==3).item()
    dtype=torch.bfloat16 if props.major>=8 else torch.float16
    a=torch.randn(256,256,device='cuda',dtype=dtype,requires_grad=True); loss=(a@a).float().square().mean(); loss.backward(); assert torch.isfinite(a.grad).all().item()
    print(f'GPU {rank}: NCCL all_reduce + {dtype} forward/backward OK',flush=True)
finally: dist.destroy_process_group()
