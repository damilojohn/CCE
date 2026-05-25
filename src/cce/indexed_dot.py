import torch
import triton
import triton.language as tl


@triton.jit
def indexed_dot(
    e_ptr,
    c_ptr,
    x_ptr,
    Out,
    B,
    D,
    V,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
):
    pid = tl.program_id(axis=0)
    n_d_chunks = tl.cdiv(D, BLOCK_D)

    # What tokens am I handling ?
    pid_b = pid // n_d_chunks
    # What blocks of D am I handling
    pid_d = pid % n_d_chunks

    # address offsets for B
    b_offsets = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    # address offsets for D
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    E_ptr = e_ptr + b_offsets[:, None] * stride_eb + d_offsets[None, :] * stride_ed

    targets = tl.load(x_ptr + b_offsets, mask=b_offsets < B)

    C_ptr = c_ptr + targets[:, None] * stride_cv + d_offsets[None, :] * stride_cd

    out_ptr = Out + b_offsets

    # load e and c tiles

    e = tl.load(
        E_ptr, mask=(b_offsets[:, None] < B) & (d_offsets[None, :] < D), other=0.0
    )

    c = tl.load(
        C_ptr, mask=(targets[:, None] < V) & (d_offsets[None, :] < D), other=0.0
    )
    # elementwise multiplication
    dot = (e * c).to(tl.float32)

    neg_dot = -tl.sum(dot, 1).to(out_ptr.dtype.element_ty)

    tl.atomic_add(out_ptr, neg_dot, mask=b_offsets < B)


def indexed_dot_kernel(
    E: torch.Tensor, C: torch.Tensor, X: torch.Tensor, BLOCK_B=16, BLOCK_D=256
):
    # get dimensions
    D = E.shape[-1]
    B = E.shape[0]
    V = C.shape[0]
    # create tensors to store dot products
    O = E.new_zeros((B), dtype=torch.float32)

    # create thread grid
    grid = (triton.cdiv(B, BLOCK_B) * triton.cdiv(D, BLOCK_D),)
    # launch kernel
    indexed_dot[grid](
        E,
        C,
        X,
        O,
        B,
        D,
        V,
        BLOCK_B,
        BLOCK_D,
        E.stride(0),
        E.stride(1),
        C.stride(0),
        C.stride(1),
    )
    return O
