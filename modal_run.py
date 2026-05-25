from typing import Literal
from fire import Fire
import structlog
import modal


LOG = structlog.stdlib.get_logger()

RunType = Literal["benchmark", "train"]

image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.12")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .apt_install("git")
    .pip_install(
        "fire",
        "torch",
        "accelerate",
        # "flash-attn",
        "triton",
        "wandb",
        "structlog",
        "datasets",
        "transformers==5.7.0",
        "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git",
        # extra_options="--only-binary=flash-attn"
    )
    .workdir("/root/cce")
    .add_local_dir("src", remote_path="/root/cce/src")
)
cce_cache_vol = modal.Volume.from_name("cce-cache", create_if_missing=True)
app = modal.App("Cut Cross Entropy")

N_GPU = 1
HOURS = 4


def run_benchmark():
    import subprocess
    import os
    import sys

    LOG.info("starting cce benchmark process....")

    process = subprocess.Popen(
        ["python", "-m", "src.benchmark"],
        env=os.environ.copy(),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    process.wait()

    if process.returncode != 0:
        LOG.error("Benchmark process failed")
        raise RuntimeError("Benchmark process failed")


def run_training(
    model_name,
    output_dir,
    per_device_train_batch_size,
    gradient_accumulation_steps,
    per_device_eval_batch_size,
    cross_entropy_impl,
    eval_strategy,
    eval_steps,
    learning_rate,
    dataloader_num_workers,
    run_name,
    report_to,
):
    from typing import cast
    import transformers
    import torch
    import wandb
    import os
    from src.training import (
        download_hf,
        make_supervised_data_module,
        patch_gemma4,
        push_to_hub,
        DataArguments,
        DATA_NAME_MAP,
        patch_options,
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "True"

    MODEL_NAME_MAP = {
        "gemma4": "google/gemma-4-E2B-it",
        "gemma2": "google/gemma-2-2b-it",
    }
    model_name = MODEL_NAME_MAP.get(model_name, model_name)
    dataset_name = DATA_NAME_MAP.get("alpaca")

    download_hf(model_name)
    download_hf(dataset_name, "dataset")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    config = transformers.AutoConfig.from_pretrained(model_name)

    data_args = DataArguments(dataset_name=dataset_name)
    data_module = make_supervised_data_module(
        tokenizer,
        data_args,
        seed=42,
        uses_system_prompt=config.model_type not in ("gemma4",),
    )

    attn_impl = "sdpa" if config.model_type != "gemma2" else "eager"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
    )
    device = torch.device("cuda", torch.cuda.current_device())
    model = model.to(device)
    model = cast(transformers.PreTrainedModel, model)
    model = patch_gemma4(model, patch_options)

    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=per_device_eval_batch_size,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        learning_rate=learning_rate,
        dataloader_num_workers=dataloader_num_workers,
        run_name=run_name,
        report_to=[report_to],
        bf16=True,
    )

    wandb.init(
        project="cut-cross-entropy",
        name=run_name,
        config={
            "model": model_name,
            "dataset": dataset_name,
            "cross_entropy_impl": cross_entropy_impl,
            "attn_impl": attn_impl,
            "learning_rate": learning_rate,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        },
    )

    class MemoryCallback(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if torch.cuda.is_available():
                wandb.log(
                    {
                        "memory/allocated_gb": torch.cuda.memory_allocated() / 1e9,
                        "memory/reserved_gb": torch.cuda.memory_reserved() / 1e9,
                        "memory/peak_allocated_gb": torch.cuda.max_memory_allocated()
                        / 1e9,
                    },
                    step=state.global_step,
                )

    trainer = transformers.Trainer(
        model,
        training_args,
        # tokenizer=tokenizer,
        callbacks=[MemoryCallback()],
        **data_module,
    )
    trainer.train()
    push_to_hub(model, tokenizer, repo_id=f"damilojohn/{run_name}")
    wandb.finish()


@app.function(
    image=image,
    gpu=f"A100-80GB:{N_GPU}",
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=HOURS * 60 * 60,
    volumes={"/data": cce_cache_vol},
)
def run(
    type: RunType,
    model_name: str,
    output_dir: str = "checkpoints",
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    per_device_eval_batch_size: int = 6,
    cross_entropy_impl: str = "cce",
    eval_strategy: str = "no",
    eval_steps: int = 1000,
    learning_rate: float = 2e-5,
    dataloader_num_workers: int = 4,
    run_name: str = "fine_tune_gemma4",
    report_to: str = "wandb",
    is_distributed: bool = False,
):
    if run_name is None:
        run_name = model_name

    if type == "benchmark":
        run_benchmark()
    elif is_distributed:
        import subprocess
        import os
        import sys

        cmd = [
            "torchrun",
            "--standalone",
            "--nproc-per-node=8",
            "--module",
            "src.training",
            "--deepspeed",
            "src/zero3.json",
            "--model_name",
            model_name,
            "--output_dir",
            output_dir,
            "--per_device_train_batch_size",
            str(per_device_train_batch_size),
            "--gradient_accumulation_steps",
            str(gradient_accumulation_steps),
            "--per_device_eval_batch_size",
            str(per_device_eval_batch_size),
            "--cross_entropy_impl",
            cross_entropy_impl,
            "--eval_strategy",
            eval_strategy,
            "--eval_steps",
            str(eval_steps),
            "--learning_rate",
            str(learning_rate),
            "--dataloader_num_workers",
            str(dataloader_num_workers),
            "--run_name",
            run_name,
            "--report_to",
            report_to,
        ]

        LOG.info("starting distributed training...", cmd=cmd)

        process = subprocess.Popen(
            cmd,
            env=os.environ.copy(),
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        process.wait()

        if process.returncode != 0:
            LOG.error("Distributed training failed", returncode=process.returncode)
            raise RuntimeError(
                f"Training process exited with code {process.returncode}"
            )
    else:
        run_training(
            model_name=model_name,
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=per_device_eval_batch_size,
            cross_entropy_impl=cross_entropy_impl,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            learning_rate=learning_rate,
            dataloader_num_workers=dataloader_num_workers,
            run_name=run_name,
            report_to=report_to,
        )


if __name__ == "__main__":
    Fire(run)
