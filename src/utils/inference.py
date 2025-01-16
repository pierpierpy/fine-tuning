import os
from typing import List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from transformers.generation.stopping_criteria import EosTokenCriteria
import torch

FAST = os.environ.get("FAST")
to_mb = 1024**2


def mistral_inference(
    model_name: str = os.path.join(FAST, "models/mistral"),
    message: str = "what is a transformer in NLP?",
    device: str = "cuda",
):
    print("loading the model")
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        cache_dir=os.path.join(FAST, ".cache"),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        device_map=device,
        cache_dir=os.path.join(FAST, ".cache"),
    )
    print("model loaded!")

    messages = [{"role": "user", "content": message}]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0], model.device


def llama_inference(
    model_name: str = os.path.join(FAST, "FTmodels/llama-7b-001b"),
    message: str = "what is a transformer in NLP?",
    device="cuda",
):
    print("loading the model")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        device_map=device,
        cache_dir=os.path.join(FAST, ".cache"),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        cache_dir=os.path.join(FAST, ".cache"),
    )
    print("model loaded!")

    input_tokens = tokenizer.encode(message, return_tensors="pt").to(device)
    output_tokens = model.generate(input_tokens, max_length=200, num_return_sequences=1)
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return output_text, model.device


def batch_inference(
    tokenized_sentences: List[List[int]],
    tokenizer: PreTrainedTokenizer = None,
    model: PreTrainedModel = None,
    device: str = "cuda",
    model_name: str = os.path.join(FAST, "FTmodels/llama-7b-001b"),
    max_length: int = 1200,
    num_return_sequences: int = 1,
):
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            device_map=device,
            cache_dir=os.path.join(FAST, ".cache"),
        )
    if model == None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            cache_dir=os.path.join(FAST, ".cache"),
        )

    # memory_consumption = torch.cuda.max_memory_allocated(device) / to_mb
    # tk_len = len(tokenized_sentences["input_ids"])
    # print(
    #     f"CHECK model on {model.device}, memory {memory_consumption}, generating on {tk_len}"
    # )

    with torch.inference_mode():
        output_tokens = model.generate(
            **tokenized_sentences,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            # stopping_criteria=[EosTokenCriteria(eos_token_id=tokenizer.eos_token_id)]
        )
    # memory_consumption = torch.cuda.max_memory_allocated(device) / to_mb
    # print(f"memory after {memory_consumption}")
    outputs_tokenized = [
        tok_out[len(tok_in) :]
        for tok_in, tok_out in zip(tokenized_sentences["input_ids"], output_tokens)
    ]

    # mask_empty_sentences = [
    #     (1 if not len(torch.unique(tok_out[len(tok_in) :])) == 1 else 0)
    #     for tok_in, tok_out in zip(tokenized_sentences["input_ids"], output_tokens)
    # ]

    num_tokens = sum([len(t) for t in outputs_tokenized])
    output_texts = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
    mask_empty_sentences = [(1 if tok_out else 0) for tok_out in output_texts]
    output_texts_masked = [
        d for d, m in zip(output_texts, mask_empty_sentences) if m == 1
    ]
    # print(f"produced {len(output_texts)} sentences in output")

    # print("----------------------------------------")
    return output_texts_masked, num_tokens, mask_empty_sentences, output_texts


def prepare_prompts(
    prompts: List[List[str]], tokenizer: PreTrainedTokenizer, batch_size=16
):
    batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
    batches_tok = []

    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                pad_to_multiple_of=8,
                add_special_tokens=False,
            ).to("cuda")
        )
    tokenizer.padding_side = "right"
    return batches_tok
