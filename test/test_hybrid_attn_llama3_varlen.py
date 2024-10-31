import random

from yunchang import (
    LongContextVarLenAttentionForLlaMa3,
    llama3_varlen_attention_sp_ulysses_ring,
    set_seq_parallel_pg,
)
import torch
import torch.distributed as dist
from flash_attn import flash_attn_varlen_func


def log(msg, a, rank0_only=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank0_only:
        if rank == 0:
            print(
                f"[Rank#0] {msg}: "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        return

    for i in range(world_size):
        if i == rank:
            if rank == 0:
                print(f"{msg}:")
            print(
                f"[Rank#{rank}] "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        dist.barrier()


if __name__ == "__main__":
    torch.random.manual_seed(0)
    random.seed(0)

    use_bwd = False
    dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 1
    nheads = 8
    d = 128
    dropout_p = 0
    causal = True
    deterministic = False

    # 8 张卡，每张卡是 q 序列长度是 529
    cu_seqlens = [0, 120, 1248, 4232]
    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    max_seqlen = (cu_seqlens_tensor[1:] - cu_seqlens_tensor[:-1]).max().item()
    total_length = cu_seqlens[-1]
    local_length = total_length // world_size
    num_seq = len(cu_seqlens) - 1
    assert cu_seqlens_tensor[-1] % world_size == 0
    assert d % 8 == 0

    # Prepare inputs
    q = torch.randn(
        total_length, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        total_length, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        total_length, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    dout = torch.randn(total_length, nheads, d, device=device, dtype=dtype)

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)
    dist.broadcast(dout, src=0)

    # 先计算原始的 attention
    out, lse, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_tensor,
        cu_seqlens_tensor,
        max_seqlen,
        max_seqlen,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )
    local_out = out[rank * local_length: (rank + 1) * local_length]

    # 计算混合 attention
    local_q = q.chunk(world_size, dim=0)[rank].detach().clone()
    local_q.requires_grad = True
    local_k = k.chunk(world_size, dim=0)[rank].detach().clone()
    local_k.requires_grad = True
    local_v = v.chunk(world_size, dim=0)[rank].detach().clone()
    local_v.requires_grad = True

    local_dout = dout.chunk(world_size, dim=0)[rank].detach().clone()

    sp_ulysses_degree = 4
    sp_ring_degree = 2
    set_seq_parallel_pg(sp_ulysses_degree, sp_ring_degree, rank, world_size)

    # longcontex = LongContextVarLenAttentionForLlaMa3()
    local_out2 = llama3_varlen_attention_sp_ulysses_ring(
        local_q,
        local_k,
        local_v,
        cu_seqlens_tensor,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        deterministic=deterministic
    )
    log('out raw', local_out, rank0_only=True)
    log('out att', local_out2, rank0_only=True)
    log("out diff", local_out - local_out2)

    out.backward(dout)
    dq = q.grad
    local_dq = dq[rank * local_length: (rank + 1) * local_length]
    dk = k.grad
    local_dk = dk[rank * local_length: (rank + 1) * local_length]
    dv = v.grad
    local_dv = dv[rank * local_length: (rank + 1) * local_length]

    local_out2.backward(local_dout)
    llama3_dq = local_q.grad
    llama3_dk = local_k.grad
    llama3_dv = local_v.grad
    log("dq diff", local_dq - llama3_dq)
    log("dk diff", local_dk - llama3_dk)
    log("dv diff", local_dv - llama3_dv)
