from typing import Dict, Any, Tuple, Union
from torch import Tensor

from transformers import PreTrainedTokenizer, AutoTokenizer, PreTrainedModel


def apply_chat_template(
    ex: Dict[str, Any], tokenizer: PreTrainedTokenizer, _eval: bool = False
):
    return {"text": formatting_function(ex=ex, tokenizer=tokenizer, _eval=_eval)}


def formatting_function(
    ex: Dict[str, Any], tokenizer: PreTrainedTokenizer, _eval: bool = False
):
    question = ex["question"]
    answer = ex["answer"]
    # context = ex["chunk"]
    messages = [
        {
            "role": "system",
            "content": "You are a cyber security specialist with many years of experience. you goal is to answer the user questions.",
        },
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]  # TODO aggiungere il contesto
    if _eval:
        messages.pop()
    return tokenizer.apply_chat_template(messages, tokenize=False)


def preprocess_logits_for_metrics(logits: Union[Tuple, Tensor]):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def load_tokenizer(
    base_model_folder: str,
    template: str = None,
    max_lenght: int = None,
    padding_side: str = "right",
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_folder, trust_remote_code=True)
    # tokenizer do not have a out of the box padding token, we have to provide it:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.model_max_length > 100_000 and max_lenght:
        tokenizer.model_max_length = max_lenght
    # all the messages are truncated or padded to max_lenght tokens
    # https://huggingface.co/docs/transformers/main/en/chat_templating
    if template:
        tokenizer.chat_template = template
    tokenizer.padding_side = padding_side
    return tokenizer


def get_model_size(model: PreTrainedModel):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return "model size: {:.3f}MB".format(size_all_mb)


### USELESS ASSETS ###
SYS_MESSAGE = """You are a cyber analyst specialist with many years of experience. Your job is to answer a <QUESTION> following this set of rules:
- The answer should be exhaustive and should use the same terminology presented in the <CONTEXT>.
- The answer should match the same language used in the chunk (<CONTEXT> in english needs an answer in english, <CONTEXT> in italian needs an answer in italian)"""
