# https://huggingface.co/blog/llama2#how-to-prompt-llama-2
# https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k?row=2
# https://www.datacamp.com/tutorial/fine-tuning-llama-2
import os
from tqdm import tqdm
from typing import Callable, List

import src.utils.fine_tuning as ft_utils

from transformers import PreTrainedTokenizer
from datasets import load_dataset, DatasetDict

FAST = os.environ.get("FAST")


def chars_token_ratio(
    dataset: DatasetDict, tokenizer: PreTrainedTokenizer, loader, nb_examples=400
):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = loader(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def load_custom_dataset(
    dataset_name: str,
    chat_template_func: Callable = None,
    subset: List[int] = [0, 100],
    field: str = "data",
    tokenizer: PreTrainedTokenizer = None,
    num_proc: int = 3,
    test_size: float = None,
    tt_split_seed: int = 235236,
    shuffle_seed: int = 235235,
    _eval: bool = False,
):
    dataset_path = os.path.join(FAST, f"data/{dataset_name}.json")
    if not os.path.exists(dataset_path):
        raise ValueError(f"no dataset at {dataset_path}")
    start, finish = subset
    dataset = (
        load_dataset(
            "json",
            name=dataset_name,
            data_files=dataset_path,
            field=field,
            num_proc=num_proc,
        )["train"]
        .shuffle(seed=shuffle_seed)
        .select(range(start, finish))
    )
    if chat_template_func:
        dataset = dataset.map(
            chat_template_func,
            num_proc=num_proc,
            fn_kwargs={"tokenizer": tokenizer, "_eval": _eval},
            desc="apply chat template",
        )
    if not test_size:
        # print(f"testing on {len(dataset)}")
        return DatasetDict(
            {"test": dataset}
        )  # if the test size is not passed, all the dataset is used to evaluate
    dataset = dataset.train_test_split(test_size=test_size, seed=tt_split_seed)
    train_data = dataset["train"]
    test_data = dataset["test"]

    print(
        f"Size of the train set: {len(train_data)}. Size of the validation set: {len(test_data)}"
    )

    return dataset
