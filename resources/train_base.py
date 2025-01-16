# MODEL PARALLEL
# the model is uploaded on each gpu in shards automatically by using auto as device for the model
##########################################################################################
import os
import debugpy
import src.utils.debug as db_utils

# host = os.getenv("HOSTNAME", "localhost")
# print(host)
# print(os.environ.get("SLURM_JOB_ID"))


# db_utils.listen(host, port=5678)
# debugpy.wait_for_client()

# run the script with python
# or
# to debug: python3 -m debugpy --listen [hostname]:5678 --wait-for-client resources/base.py
##########################################################################################

import src.utils.fine_tuning as ft_utils
import src.utils.dataset as ds_utils

import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments

# from accelerate import PartialState
from datetime import datetime

FAST = os.environ.get("FAST")
device_map = "auto"
base_model_id = "llama"
ft_model_id = "llama"

base_model_folder = os.path.join(FAST, "models", base_model_id)
lora_alpha = 8
lora_dropout = 0.1
dataset_name = "dataset_002"
lora_r = 32
time_signature = datetime.now().strftime("%b_%d_%Y_%H_%M")
computed_ft_model_id: str = f"{ft_model_id}_{time_signature}_{dataset_name}"
ft_model_folder: str = os.path.join(FAST, "FTmodels", computed_ft_model_id)
tokenizer = ft_utils.load_tokenizer(base_model_folder, template=ft_utils.template)

dataset = ds_utils.load_custom_dataset(
    dataset_name=dataset_name,
    subset=[0, 100],
    field="data",
    num_proc=5,
    test_size=0.1,
    shuffle_seed=235236,
    tt_split_seed=2352353,
    tokenizer=tokenizer,
)

bnb_configs = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model_folder,
    quantization_config=bnb_configs,
    cache_dir="",
    use_cache=False,
    device_map=device_map,
    cache_dir=os.path.join(FAST, ".cache"),
)

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
    modules_to_save=[
        "embed_tokens",
        "input_layernorm",
        "post_attention_layernorm",
        "norm",
    ],
)
max_seq_length = 512
output_dir = ft_model_folder
per_device_train_batch_size = 8
gradient_accumulation_steps = 2
optim = "adamw_torch"
save_steps = 10
logging_steps = 1
learning_rate = 2e-4
max_grad_norm = 0.3
# max_steps = 1  # Approx the size of guanaco at bs 8, ga 2, 2 GPUs.
warmup_ratio = 0.1
lr_scheduler_type = "cosine"
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    bf16=True,
    num_train_epochs=25,
    max_grad_norm=max_grad_norm,
    # max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},  # must be false for DDP,
)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)
