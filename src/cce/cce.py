from typing import Optional
import torch
from cce.indexed_dot import indexed_dot_kernel
from cce.logsumexp import lse_kernel


class CutCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        e: torch.Tensor,
        c: torch.Tensor,
        x: torch.Tensor,
        BLOCK_B: Optional[int],
        BLOCK_D: Optional[int],
        BLOCK_V: Optional[int]
    ):

        # call indexed dot product kernel
        nll = indexed_dot_kernel(e, c, x, BLOCK_B, BLOCK_D)
        # call log sum exp kernel
        lse = lse_kernel(e, c, x, BLOCK_B, BLOCK_D, BLOCK_V)
        # add result
        loss = nll.add_(lse)

        # save whatever we need for the backward pass
        ctx.save_for_backward(e, c, x, lse)

        return loss

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        pass


def linear_cut_cross_entropy(
        e: torch.Tensor,
        c: torch.Tensor,
        x: torch.Tensor,
        BLOCK_B: Optional[int],
        BLOCK_D: Optional[int],
        BLOCK_V: Optional[int]     
):
    loss = CutCrossEntropy(e, c, x)
    return loss