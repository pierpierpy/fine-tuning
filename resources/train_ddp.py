# DISTRIBUTED DATA PARALLEL
#
##########################################################################################
# import os
# import debugpy
# import src.utils.debug as db_utils

# host = os.getenv("HOSTNAME", "localhost")
# print(host)
# print(os.environ.get("SLURM_JOB_ID"))


# db_utils.listen(host, port=5678)
# debugpy.wait_for_client()

# to debug:
# accelerate launch --multi_gpu --num_processes=2 train_ddp.py

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

from accelerate import PartialState
from datetime import datetime
import os

device_map = "DDP"  # for DDP and running with `accelerate launch test_sft.py`

if device_map == "DDP":
    device_string = PartialState().process_index
    device_map = {"": device_string}
FAST = os.environ.get("FAST")
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
max_steps = 1  # Approx the size of guanaco at bs 8, ga 2, 2 GPUs.
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
    gradient_checkpointing=True,  # is a way to reduce the vram to store the activation during the forward pass that are used to calculate the gradients
    # the higher the input sequence, and the higher the number of batches the more the activation is heavy, to avoid storing all the activation
    # we can set this to True, in this way we store some checkpoints of the activation and not the whole activation, the missing parts are than recalculated
    # on the fly during the backward prop as they are needed. it is going to increase the number of calcualtions in case we have constraints with VRAM
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },  # this is just a better way to fill the missing parts of the activations with gradient checkpointing. SET THIS TO False for DDP
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
