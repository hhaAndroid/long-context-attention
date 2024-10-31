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


from typing import Tuple, Optional
import torch
import torch.nn.functional as F


@torch.jit.script
def _update_out_and_lse(
        out: torch.Tensor,
        lse: torch.Tensor,
        block_out: torch.Tensor,
        block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
        out: Optional[torch.Tensor],
        lse: Optional[torch.Tensor],
        block_out: torch.Tensor,
        block_lse: torch.Tensor,
        slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 1
    seqlen = 3816
    nheads = 5
    d = 128
    dropout_p = 0
    causal = True
    deterministic = False

    assert seqlen % world_size == 0
    assert d % 8 == 0

    q = torch.randn(
        batch_size, 1, nheads, d, device=device, dtype=dtype, requires_grad=True
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

    local_q = q
    local_k = k.chunk(world_size, dim=1)[rank]
    local_v = v.chunk(world_size, dim=1)[rank]
    local_dout = dout.chunk(world_size, dim=1)[rank]

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# forward:")
        print("#" * 30)

    out, lse, _ = flash_attn_func(
        q, k, v,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

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

    # 由于累加计算误差，我们只需要最后一个 rank 返回值
    dist.broadcast(ring_out, src=world_size - 1)
    dist.broadcast(ring_lse, src=world_size - 1)
    log("out", out, rank0_only=True)
    log("ring_out", ring_out, rank0_only=True)
    log("out diff", out - ring_out, rank0_only=True)
    log("lse diff", lse - ring_lse, rank0_only=True)

    # 由于 q=1，因此有如下简化算法
    out1, lse, _ = flash_attn_func(
        local_q, local_k, local_v,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )
    out_list = [
        torch.empty_like(out1, device=out.device, dtype=out.dtype)
        for _ in range(world_size)
    ]
    dist.all_gather(out_list, out1)
    out_lse = [
        torch.empty_like(lse, device=lse.device, dtype=lse.dtype)
        for _ in range(world_size)
    ]
    dist.all_gather(out_lse, lse)

    # 计算修正后的 out
    new_out = None
    new_lse = None
    for i in reversed(range(world_size)):
        new_out, new_lse = update_out_and_lse(new_out, new_lse, out_list[i], out_lse[i])
    new_out = new_out.to(q.dtype)
    log("new out diff", out - new_out, rank0_only=True)

