import os

from flash_attn import flash_attn_func
import torch
import torch.distributed as dist
from yunchang import ring_flash_attn_func, ring_flash_attn_inference_func


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
    print(world_size)
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

    # 模拟 kvcache 下的 ring flash attn
    # q 序列长度始终是1，kv 序列长度假设一共是 seqlen，那么在下一次 forward 时候，kv 长度是 seqlen+1，而且是拼接在最后
    q = torch.randn(
        batch_size, 1, nheads, d, device=device, dtype=dtype
    )

    # 完整 kv cache
    k = torch.randn(
        batch_size, seqlen + 1, nheads, d, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, seqlen + 1, nheads, d, device=device, dtype=dtype
    )
    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)

    dout = torch.randn(batch_size, 1, nheads, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    # 切分方式比较特别
    local_q = q
    local_k = k[:, :seqlen, ...].chunk(world_size, dim=1)[rank]
    local_v = v[:, :seqlen, ...].chunk(world_size, dim=1)[rank]
    if rank == world_size - 1:
        local_k = torch.cat([local_k, k[:, seqlen:, ...]], dim=1)
        local_v = torch.cat([local_v, v[:, seqlen:, ...]], dim=1)
    print(f'=xxx=={rank, local_q.shape, local_k.shape}')

    local_dout = dout

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

    new_out = ring_flash_attn_inference_func(local_q, local_k, local_v,
                                             dropout_p=dropout_p,
                                             causal=causal,
                                             window_size=(-1, -1),
                                             alibi_slopes=None,
                                             deterministic=deterministic,
                                             return_attn_probs=True)
    log("new out diff", out - new_out, rank0_only=True)

