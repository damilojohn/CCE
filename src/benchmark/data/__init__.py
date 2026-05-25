import functools
from collections.abc import Callable

from .data import Data
from .models import generate_test_data_otf, load_model
from .randn import generate as randn_generate

generators: dict[str, Callable[..., Data]] = {
    "llama3": functools.partial(
        generate_test_data_otf,
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ),
    "llama3.2-1": functools.partial(
        generate_test_data_otf,
        "meta-llama/Llama-3.2-1B-Instruct",
    ),
    "llama3.2-3": functools.partial(
        generate_test_data_otf,
        "meta-llama/Llama-3.2-3B-Instruct",
    ),
    "llama3-70": functools.partial(
        generate_test_data_otf,
        "meta-llama/Meta-Llama-3-70B-Instruct",
    ),
    "gemma2": functools.partial(generate_test_data_otf, "google/gemma-2-2b-it"),
    "gemma2-9": functools.partial(generate_test_data_otf, "google/gemma-2-9b-it"),
    "gemma2-27": functools.partial(generate_test_data_otf, "google/gemma-2-27b-it"),
    "gemma4": functools.partial(generate_test_data_otf, "google/gemma-4-E2B-it"),
    "phi3.5": functools.partial(
        generate_test_data_otf, "microsoft/Phi-3.5-mini-instruct"
    ),
    "mistral-nemo": functools.partial(
        generate_test_data_otf, "mistralai/Mistral-Nemo-Instruct-2407"
    ),
}

generators = generators | {
    f"{k}-invalids": functools.partial(v, keep_invalids=True)
    for k, v in generators.items()
}

generators["randn"] = randn_generate


def generator(name: str) -> Callable[..., Data]:
    if name not in generators:
        raise ValueError(f"Data generator {name!r} not found.")

    load_model.cache_clear()
    return generators[name]
