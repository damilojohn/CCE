import triton
import triton.language as tl
import torch


def log_sum_exp_kernel(
    E_ptr,  # Hidden states 
    C_ptr,  # classifer head  
):

    pid = tl.program_id(axis=0)