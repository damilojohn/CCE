import functools
import gc
import time
from dataclasses import asdict
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from fire import Fire

from cut_cross_entropy import linear_cross_entropy
from cut_cross_entropy.constants import IGNORE_INDEX

from . import data, memory


def baseline(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    softcap: float | None = None,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
) -> torch.Tensor:
    logits = e @ c.T

    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap

    return F.cross_entropy(
        logits.float(), targets, ignore_index=ignore_index, reduction=reduction
    )


def clear_grad_fn(E, C, *args, **kwargs):
    E.grad = C.grad = None


def benchmark(
    methods: list[str] | str | None = None,
    test_data: list[str] | str | None = "mistral-nemo",
    n_iteration: int = 50,
    n_rep: int = 1,
    dtype: str = "bfloat16",
    output: str | None = None,
    kinds: list[str] | str | None = "loss-fw-bw",
    softcap: float | None = None,
):
    torch.set_float32_matmul_precision("high")

    if methods is None:
        methods = ["cce", "torch_compile", "baseline"]
    elif isinstance(methods, str):
        methods = methods.split(",")

    if kinds is None:
        kinds = ["loss-fw", "loss-bw", "loss-fw-bw"]
    elif isinstance(kinds, str):
        kinds = kinds.split(",")

    if test_data is None:
        test_data = [
            "gemma4",
            # "llama3",
            "mistral-nemo",
            "phi3.5",
        ]
    elif isinstance(test_data, str):
        test_data = test_data.split(",")

    dtype = getattr(torch, dtype)

    all_stats = []

    for this_test_data in tqdm.tqdm(
        test_data, desc="Data source", disable=len(test_data) == 1
    ):
        gen = data.generator(this_test_data)
        for rep in tqdm.trange(n_rep + 1, desc="Repetition"):
            D = gen(dtype=dtype)
            for kind in tqdm.tqdm(
                kinds, desc="Benchmark kind", disable=len(kinds) == 1
            ):
                E, C, T = D.embedding, D.classifier, D.targets

                this_softcap = softcap if softcap is not None else D.softcap

                kwargs: dict[str, Any] = {"softcap": this_softcap}
                if kind == "loss-fw":
                    E.requires_grad_(False)
                    C.requires_grad_(False)
                    args = (E, C, T)
                elif kind in {"loss-bw", "loss-fw-bw"}:
                    E.requires_grad_(True)
                    C.requires_grad_(True)

                    args = (E, C, T)
                    kwargs["backward"] = True
                    kwargs["forward"] = kind == "loss-fw-bw"
                    kwargs["pre_fn"] = clear_grad_fn
                else:
                    raise ValueError(f"Unknown kind {kind=}")

                for m in tqdm.tqdm(methods, desc="Method", leave=False):
                    if m in "liger" and kind.startswith("lse"):
                        continue
                    print(f"Loss calculated with {m}")

                    # warmup (it==0)
                    stats = memory.Stats.measure(
                        (
                            baseline
                            if m == "baseline"
                            else functools.partial(linear_cross_entropy, impl=m)
                        ),
                        *args,
                        n_iteration=n_iteration if rep > 0 else 1,
                        **kwargs,
                    )

                    if rep > 0 or n_rep == 0:
                        this_stats = {
                            "method": m,
                            "kind": kind,
                        } | asdict(stats)

                        this_stats["test_data"] = this_test_data

                        all_stats.append(this_stats)

                        torch.cuda.synchronize()
                        time.sleep(1)
                        gc.collect()
                        torch.cuda.empty_cache()
                        time.sleep(1)

    if not all_stats:
        print("No stats collected.")
        return

    df = pd.DataFrame(all_stats)

    for (test_data, kind), group in df.groupby(["test_data", "kind"]):
        print(f"\n{'=' * 60}")
        print(f"  {test_data}  |  {kind}")
        print(f"{'=' * 60}")
        print(f"{'Method':<20} {'Time (ms)':>12} {'Memory (MB)':>14}")
        print(f"{'-' * 20} {'-' * 12} {'-' * 14}")
        for _, row in group.sort_values("runtime_ms").iterrows():
            print(
                f"{row['method']:<20} {row['runtime_ms']:>12.2f} {row['op_mem_mb']:>14.1f}"
            )

    if output is not None:
        df.to_csv(output, index=False)
        print(f"\nResults saved to {output}")


if __name__ == "__main__":
    Fire(benchmark)
