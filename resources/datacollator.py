import yaml
import os

import src.utils.dataset as ds_utils
import src.utils.fine_tuning as ft_utils

from transformers import PreTrainedTokenizer
from trl import DataCollatorForCompletionOnlyLM
from datasets import DatasetDict

FAST = os.environ.get("FAST")

with open(
    "/leonardo/home/userexternal/pdipasqu/hpc-leonardo-scripts/configs/TRAINconfig.yaml",
    "r",
) as yaml_file:
    kwargs = yaml.load(yaml_file, Loader=yaml.FullLoader)
base_configs = kwargs["base_configs"]

base_model_folder: str = os.path.join(FAST, "models", base_configs["base_model_id"])


tokenizer: PreTrainedTokenizer = ft_utils.load_tokenizer(
    **kwargs["tokenizer_configs"],
    **dict(base_model_folder=base_model_folder),
)

dataset: DatasetDict = ds_utils.load_custom_dataset(
    **kwargs["dataset_configs"],
    **dict(
        chat_template_func=ft_utils.apply_chat_template,
        tokenizer=tokenizer,
    ),
)
datacollator: DataCollatorForCompletionOnlyLM = (
    (
        DataCollatorForCompletionOnlyLM(
            **kwargs["data_collator"]["config"],
            **dict(
                tokenizer=tokenizer,
            ),
        )
    )
    if kwargs["data_collator"]["do_data_collator"]
    else None
)
# use the following to see the collator at work. We see how the tokens from the user question and system message are ignored with the "ignore index" that is -100

example = dataset["train"]["text"][0]
tokenized_example = tokenizer(example)["input_ids"]
datacollator.torch_call([tokenized_example])  # --> takes the data in batch


# try this to show the ignored tokens:
example = dataset["train"]["text"][0]
tokenized_example = tokenizer(example)["input_ids"]
labels = datacollator.torch_call([tokenized_example])["labels"][
    0
]  # take the tokens with the ignore index
filtered_labels = [
    label for label in labels if label != -100
]  # filter the ignore index
decoded_text = tokenizer.decode(
    filtered_labels
)  # decode the tokens of the assistant response
decoded_text
