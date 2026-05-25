from types import MethodType
import enum
from enum import auto
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, cast
import os
import wandb
import structlog

import transformers
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from typing import Optional
import datasets
from huggingface_hub import snapshot_download
from cut_cross_entropy import linear_cross_entropy
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gemma4 import modeling_gemma4


LOG = structlog.stdlib.get_logger()


@dataclass
class PatchOptions:
    impl: str
    reduction: str


_PATCH_OPTS: PatchOptions | None = None


class LinearCrossEntropyImpl(enum.IntEnum):
    CCE = auto()
    TORCH_COMPILE = auto()


def cce_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: int = 0,
    pixel_values: torch.FloatTensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    input_features: torch.FloatTensor | None = None,
    input_features_mask: torch.Tensor | None = None,
    past_key_values: transformers.cache_utils.Cache | None = None,
    mm_token_type_ids: torch.LongTensor | None = None,
    image_position_ids: torch.LongTensor | None = None,
    video_position_ids: torch.LongTensor | None = None,
    **kwargs,
):
    logits = None
    loss = None
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        input_features=input_features,
        input_features_mask=input_features_mask,
        mm_token_type_ids=mm_token_type_ids,
        image_position_ids=image_position_ids,
        video_position_ids=video_position_ids,
    )
    hidden_states = outputs.last_hidden_state
    if labels is not None and _PATCH_OPTS is not None:
        if _PATCH_OPTS.impl == "cce":
            loss = linear_cross_entropy(
                hidden_states,
                self.lm_head.weight,
                labels.to(hidden_states.device),
                softcap=None,
                shift=True,
                impl=_PATCH_OPTS.impl,
                reduction=_PATCH_OPTS.reduction,
            )
        else:
            logits = self.lm_head(hidden_states[:, :, :])

            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        # past_key_values=outputs.past_key_values,
        hidden_states=hidden_states,
        # attentions=outputs.attentions,
    )


def patch_gemma4(transformers_model, patch_options: PatchOptions):
    global _PATCH_OPTS

    _PATCH_OPTS = patch_options
    assert isinstance(transformers_model, transformers.PreTrainedModel), (
        "Invalid model object"
    )
    assert isinstance(
        transformers_model, modeling_gemma4.Gemma4ForConditionalGeneration
    ), "Invalid Transformers Model type"

    transformers_model.forward = MethodType(cce_forward, transformers_model)
    return transformers_model


LCE_IMPL_DEFAULT = LinearCrossEntropyImpl.CCE

patch_options = PatchOptions(impl="cce", reduction="mean")
device = "cuda" if torch.cuda.is_available else "cpu"

DATA_NAME_MAP = {"alpaca": "yahma/alpaca-cleaned"}

IGNORE_INDEX = -100
SYSTEM_PROMPT = "You are a helpful AI assistant."
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
    ),
}


@dataclass
class ModelArguments:
    model_name: str
    attn_impl: str | None = None
    cross_entropy_impl: str = "cce"


@dataclass
class DataArguments:
    dataset_name: str = "alpaca"
    sequence_length: int = 512


def download_hf(name: str, repo_type: str = "model"):
    # if not Path(name).exists():
    #     subprocess.check_call(
    #         [
    #             "hf",
    #             "download",
    #             "--exclude=original/*",
    #             f"--repo-type={repo_type}",
    #             name,
    #         ]
    #     )

    if Path(name).exists():
        return str(Path(name))
    # Cache under /data so Modal Volume persists across runs.
    cache_dir = Path("/data/hf")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Mirrors your old CLI behavior: skip original/* artifacts.
    local_path = snapshot_download(
        repo_id=name,
        repo_type=repo_type,  # "model" or "dataset"
        local_dir=str(cache_dir / name.replace("/", "__")),
        local_dir_use_symlinks=False,
        ignore_patterns=["original/*"],
        token=os.environ.get("HF_TOKEN"),  # from your hf-secret
        resume_download=True,
    )
    return local_path


def preprocess(
    source: str,
    target: str,
    tokenizer: transformers.PreTrainedTokenizer,
    uses_system_prompt: bool = True,
) -> dict:
    """Preprocess the data by tokenizing."""
    if uses_system_prompt:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
    else:
        messages = []

    messages.extend(
        (
            {"role": "user", "content": source},
            {"role": "assistant", "content": target},
        )
    )
    tokenization = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        return_dict=True,
    )
    input_ids = torch.as_tensor(tokenization["input_ids"])

    target_ids = tokenizer.encode(
        target, add_special_tokens=False, return_tensors="pt"
    )[0]

    labels = input_ids.clone()
    for offset in reversed(range(0, len(input_ids) - len(target_ids))):
        if (labels[offset : offset + len(target_ids)] == target_ids).all():
            labels[0:offset] = IGNORE_INDEX
            break

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_args: DataArguments,
        seed: int,
        tokenizer: transformers.PreTrainedTokenizer,
        split: str = "train",
        uses_system_prompt: bool = True,
    ):
        super().__init__()
        self.dataset = datasets.load_dataset(data_args.dataset_name, split="train")
        self.tokenizer = tokenizer
        self.uses_system_prompt = uses_system_prompt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        element = self.dataset[i]
        if element["input"] == "":
            prompt_template = PROMPT_DICT["prompt_no_input"]
        else:
            prompt_template = PROMPT_DICT["prompt_input"]

        source = prompt_template.format_map(element)
        target = element["output"]

        return preprocess(source, target, self.tokenizer, self.uses_system_prompt)


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    pad_token_id: int | None
    padding_side: str

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        max_len = max(len(v) for v in input_ids)
        assert self.pad_token_id is not None
        padded_input_ids = torch.full((len(input_ids), max_len), self.pad_token_id)
        padded_labels = torch.full((len(input_ids), max_len), IGNORE_INDEX)
        position_ids = (
            torch.arange(0, max_len, dtype=torch.int64)
            .view(1, -1)
            .repeat(len(input_ids), 1)
        )

        for i, (inp, lbl) in enumerate(zip(input_ids, labels, strict=True)):
            if self.padding_side == "right":
                slc = slice(len(inp))
            else:
                slc = slice(-len(inp), None)

            padded_input_ids[i, slc] = inp
            padded_labels[i, slc] = lbl
            position_ids[i, slc] -= position_ids[i, slc][0].item()

        return dict(
            input_ids=padded_input_ids,
            labels=padded_labels,
            attention_mask=padded_input_ids.ne(self.pad_token_id),
            position_ids=position_ids,
        )


def push_to_hub(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    repo_id: str,
    private: bool = True,
):
    token = os.environ.get("HF_TOKEN")
    LOG.info("pushing model to hub", repo_id=repo_id)
    model.push_to_hub(repo_id, token=token, private=private)
    tokenizer.push_to_hub(repo_id, token=token, private=private)
    LOG.info("push complete", repo_id=repo_id)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    seed,
    uses_system_prompt: bool = True,
) -> dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        data_args,
        seed=seed,
        tokenizer=tokenizer,
        uses_system_prompt=uses_system_prompt,
    )
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer.pad_token_id, tokenizer.padding_side
    )
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = False
    torch_compile: bool = False
    fp16: bool = False
    bf16: bool = True
    tf32: bool = True
    gradient_checkpoint: bool = True
    logging_strategy: str = "steps"
    logging_steps: int = 1
    warmup_ratio: float = 0.05
    dataloader_num_workers: int = 12
    dataloader_pin_memory: bool = True
    save_strategy: str = "no"
    save_steps: int = 400
    save_total_limit: int = 3
    num_train_epochs: float = 1.0
    gradient_checkpoint_kwargs: dict[str, Any] = field(
        default_factory=lambda: dict(use_reentrant=True)
    )


if __name__ == "__main__":
    MODEL_NAME_MAP = {
        "gemma4": "google/gemma-4-E2B-it",
        "gemma2": "google/gemma-2-2b-it",
    }

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args = cast(TrainingArguments, training_args)
    model_args = cast(ModelArguments, model_args)
    data_args = cast(DataArguments, data_args)

    model_args.model_name = MODEL_NAME_MAP[model_args.model_name]

    data_args.dataset_name = DATA_NAME_MAP[data_args.dataset_name]
    # download data AND model

    if torch.distributed.is_initialized():
        if training_args.local_rank == 0:
            download_hf(model_args.model_name)
            download_hf(data_args.dataset_name, "dataset")

    # preprocess data
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name, use_fast=True
    )
    config = transformers.AutoConfig.from_pretrained(model_args.model_name)

    data_module = make_supervised_data_module(
        tokenizer,
        data_args,
        training_args.seed,
        uses_system_prompt=config.model_type not in ("gemma4",),
    )

    # setup training

    attn_impl = model_args.attn_impl
    if attn_impl is None:
        attn_impl = "eager" if config.model_type in ("gemma2", "gemma4") else "sdpa"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
    )
    device = torch.device("cuda", torch.cuda.current_device())
    model = model.to(device)
    model = cast(transformers.PreTrainedModel, model)
    model = patch_gemma4(model, patch_options)

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    wandb.init(
        project="cut-cross-entropy",
        config={
            "model": model_args.model_name,
            "dataset": data_args.dataset_name,
            "cross_entropy_impl": model_args.cross_entropy_impl,
            "sequence_length": data_args.sequence_length,
            "attn_impl": attn_impl,
            "bf16": training_args.bf16,
            "learning_rate": training_args.learning_rate,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "num_train_epochs": training_args.num_train_epochs,
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

    training_args.report_to = ["wandb"]

    trainer = transformers.Trainer(
        model,
        training_args,
        tokenizer=tokenizer,
        callbacks=[MemoryCallback()],
        **data_module,
    )
    trainer.train()

    if torch.distributed.is_initialized() and training_args.local_rank != 0:
        pass  # only rank 0 pushes
    else:
        push_to_hub(model, tokenizer, repo_id=f"damilojohn/{run_name}")

    wandb.finish()
