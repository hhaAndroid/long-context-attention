import os

from flash_attn import flash_attn_func
import torch
import torch.distributed as dist
from yunchang import ring_flash_attn_func


def log(msg, a, rank0_only=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank0_only:
        if rank == 0:
            print(
                f"{msg}: "
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
                f"[{rank}] "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        dist.barrier()


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 1
    seqlen = 4
    nheads = 16
    d = 128
    dropout_p = 0
    causal = True
    deterministic = False

    assert seqlen % world_size == 0
    assert d % 8 == 0

    for i in range(1000):
        seqlen = i * world_size * 16 + 4
        if rank == 0:
            print(f'=========================={seqlen}==========================')
        q = torch.randn(
            batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
        )
        k = torch.randn(
            batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
        )
        v = torch.randn(
            batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
        )
        dist.broadcast(q, src=0)
        dist.broadcast(k, src=0)
        dist.broadcast(v, src=0)

        dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
        dist.broadcast(dout, src=0)

        local_q = q.chunk(world_size, dim=1)[rank].detach().clone()
        local_q.requires_grad = True
        local_k = k.chunk(world_size, dim=1)[rank].detach().clone()
        local_k.requires_grad = True
        local_v = v.chunk(world_size, dim=1)[rank].detach().clone()
        local_v.requires_grad = True
        local_dout = dout.chunk(world_size, dim=1)[rank].detach().clone()

        dist.barrier()

        out, lse, _ = flash_attn_func(
            q, k, v,
            dropout_p=dropout_p,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=deterministic,
            return_attn_probs=True,
        )

        local_out = out.chunk(world_size, dim=1)[rank]
        local_lse = lse.chunk(world_size, dim=-1)[rank]

        fn = ring_flash_attn_func

        ring_out, ring_lse, _ = fn(
            local_q, local_k, local_v,
            dropout_p=dropout_p,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=deterministic,
            return_attn_probs=True,
        )

        out.backward(dout)
        dq = q.grad
        local_dq = dq.chunk(world_size, dim=1)[rank]
        dk = k.grad
        local_dk = dk.chunk(world_size, dim=1)[rank]
        dv = v.grad
        local_dv = dv.chunk(world_size, dim=1)[rank]

        ring_out.backward(local_dout)
        ring_dq = local_q.grad
        ring_dk = local_k.grad
        ring_dv = local_v.grad

        forward_max = torch.max(torch.abs(local_out - ring_out))
        dist.all_reduce(forward_max, op=dist.ReduceOp.MAX)

        bws_max_dq = torch.max(torch.abs(local_dq - ring_dq))
        dist.all_reduce(bws_max_dq, op=dist.ReduceOp.MAX)
        bws_max_dk = torch.max(torch.abs(local_dk - ring_dk))
        dist.all_reduce(bws_max_dk, op=dist.ReduceOp.MAX)
        bws_max_dv = torch.max(torch.abs(local_dv - ring_dv))
        dist.all_reduce(bws_max_dv, op=dist.ReduceOp.MAX)
        bwd_max = max(bws_max_dq.item(), bws_max_dk.item(), bws_max_dv.item())

        if rank == 0:
            print(f'forward_max: {forward_max}, bwd_max: {bwd_max}')
        # log("load_o", local_out)
        # log("load_dq", local_dq)
        # log("load_dk", local_dk)
        # log("load_dv", local_dv)
        # log("ring_dq", ring_dq)
        # log("ring_dk", ring_dk)
        # log("ring_dv", ring_dv)
        # log("out diff", local_out - ring_out)
        # log("dq diff", local_dq - ring_dq)
        # log("dk diff", local_dk - ring_dk)
        # log("dv diff", local_dv - ring_dv)
