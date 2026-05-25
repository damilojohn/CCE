import triton
import triton.language as tl
import torch
from triton.language.extra import libdevice as tl_libdevice


@triton.jit
def tl_log1p(a: tl.tensor) -> tl.tensor:
    return tl_libdevice.log1p(a)


@triton.jit
def tl_logaddexp(a, b) -> tl.tensor:
    minx = tl.minimum(a, b)
    mx = tl.maximum(a, b)
    return tl_log1p(tl.exp(minx - mx)) + mx


@triton.jit
def _lse(
    E,
    C,
    X,
    LSE,
    locks,
    B,
    D,
    V,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_B: tl.constexpr,
    num_locks,
    stride_eb,
    stride_ed,
    stride_cd,
    stride_cv,
    stride_lse_b,
):
    pid = tl.program_id(axis=0)
    num_v_chunks = tl.cdiv(V, BLOCK_V)
    pid_b = pid // num_v_chunks
    pid_v = pid % num_v_chunks

    # address offsets for b
    b_offsets = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    # address offsets for v
    v_offsets = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)

    accum = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)

    # reduction over D
    for d in range(0, D, BLOCK_D):
        d_offsets = d + tl.arange(0, BLOCK_D)

        # load E tile
        e = tl.load(
            E + b_offsets[:, None] * stride_eb + d_offsets[None, :] * stride_ed,
            mask=d_offsets[None, :] < D,
            other=0.0,
        )
        # load C tile
        c = tl.load(
            C + v_offsets[None, :] * stride_cv + d_offsets[:, None] * stride_cd,
            mask=d_offsets[:, None] < D,
            other=0.0,
        )
        # accumulate dot products
        accum = tl.dot(e, c, accum)

    # masks across V dimension to ensure we don't load invalid addresses
    v_mask = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)) < V
    o_mask = b_offsets < B
    logits = tl.where(v_mask[None, :], accum, -float("inf"))

    # max logit for this VB block
    this_max = tl.max(logits, axis=1)
    exp = tl.exp(logits - this_max[:, None])

    # log sum exp for this VB_BLOCK
    this_lse = this_max + tl.log(tl.sum(exp, axis=1))

    lse_ptrs = LSE + (b_offsets * stride_lse_b)
    # Acquire spin lock
    this_locks = locks + (pid_b // (tl.cdiv(B, BLOCK_B * num_locks)))

    while tl.atomic_cas(this_locks, 0, 1) == 1:
        pass

    # load LSE Computed by other thread blocks
    lse = tl.load(lse_ptrs, mask=o_mask, other=0.0, eviction_policy="evict_last")

    # log add exp
    lse = tl_logaddexp(lse, this_lse)

    # write back to HBM
    tl.store(lse_ptrs, lse, o_mask, eviction_policy="evict_last")

    # Release lock
    tl.atomic_xchg(this_locks, 0)


def lse_kernel(
    E: torch.Tensor,
    C: torch.Tensor,
    X: torch.Tensor,
    BLOCK_B=16,
    BLOCK_D=128,
    BLOCK_V=128,
):
    B = E.shape[0]
    D = E.shape[-1]
    V = C.shape[0]

    lse = E.new_full((B,), -float("inf"), dtype=torch.float32)

    # get locks
    locks = E.new_full(
        (triton.cdiv(B, 128),),
        0,
        dtype=torch.uint32,
    )

    grid = (triton.cdiv(B, BLOCK_B) * triton.cdiv(V, BLOCK_V),)

    _lse[grid](
        E,
        C,
        X,
        lse,
        locks,
        B,
        D,
        V,
        BLOCK_V,
        BLOCK_D,
        BLOCK_B,
        locks.stride(0),
        E.stride(0),
        E.stride(1),
        C.stride(1),
        C.stride(0),
        lse.stride(0),
    )

    return lse
