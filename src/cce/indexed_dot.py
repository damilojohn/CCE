import triton
import triton.language as tl
import torch


@triton.jit
def indexed_dot_kernel(
    E_ptr,  # Hidden states
    C_ptr,  # Language Modelling Classifier Head
    x_ptr,  # target tokens
    O_ptr,  # where are we writing dot products to?
    N,  # sequence_length
    D,  # Hidden_Dim
    V,  # Vocab size
    BLOCK_N: tl.constexpr,  # Block size across x, how many tokens is this block attending to
    BLOCK_D: tl.constexpr,  # how much of E are we loading at once across the D dimension
    stride_en,
    stride_ed,
    stride_cv,
    stride_cd,
):
    pid = tl.program_id(axis=0)  # what tokens am I handling in the input sequence?

    x_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)  # computing address offsets
    x_mask = x_offsets < N

    # loading token indices

    x = tl.load(x_ptr + x_offsets, mask=x_mask, other=0) 
    # temporary tensor to store dot products for NB tokens in SRAM
    o = tl.zeros([BLOCK_N], dtype=tl.float32)
    # reduction over D
    for d in range(0, D, BLOCK_D):
        d_offsets = d + tl.arange(0, BLOCK_D)
        # loading E tile
        e = tl.load(
            E_ptr + x_offsets[:, None] * stride_en + d_offsets[None, :] * stride_ed,
            mask=(x_offsets[:, None] < N) & (d_offsets[None, :] < D),
            other=0.0
        )
        # loading C tile
        c = tl.load(
            C_ptr + x[:, None]*stride_cv + d_offsets[None, :] * stride_cd,
            mask=(x[:,None] < V) & (d_offsets[None, :] < D),
            other=0.0
        )
        # accumulating
        o += tl.sum((e*c).to(tl.float32), axis=1)
    # writing back to global memory
    tl.store(O_ptr + x_offsets, -o, mask=x_mask)
