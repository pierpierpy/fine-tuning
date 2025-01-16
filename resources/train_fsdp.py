# FULLY SHARDER DATA PARELLEL
#
##########################################################################################
import os
import debugpy
import src.utils.debug as db_utils

host = os.getenv("HOSTNAME", "localhost")
print(host)
print(os.environ.get("SLURM_JOB_ID"))


db_utils.listen(host, port=5678)
debugpy.wait_for_client()

# to debug:
# accelerate launch --multi_gpu --num_processes=2 train_ddp.py

##########################################################################################


# Based on a script from: https://github.com/huggingface/trl/issues/1303
# Run this with FSDP with "accelerate launch test_scripts/test_fsdp.py" after having run "accelerate config"
# YOU MUST RUN "accelerate config" before running this script. See the README.md for options to select.
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments

import src.utils.fine_tuning as ft_utils
import src.utils.dataset as ds_utils
from datetime import datetime

device_map = "FSDP"  # for FSDP and running with `accelerate launch test_sft.py`
FAST = os.environ.get("FAST")
dataset_name = "dataset_002"

base_model_id = "llama"
ft_model_id = "llama"
time_signature = datetime.now().strftime("%b_%d_%Y_%H_%M")
computed_ft_model_id: str = f"{ft_model_id}_{time_signature}_{dataset_name}"
ft_model_folder: str = os.path.join(FAST, "FTmodels", computed_ft_model_id)

base_model_folder = os.path.join(FAST, "models", base_model_id)
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


model = AutoModelForCausalLM.from_pretrained(
    base_model_folder,
    trust_remote_code=True,
    cache_dir="",
    use_cache=False,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    cache_dir=os.path.join(FAST, ".cache"),
)

# PEFT config
lora_alpha = 8
lora_dropout = 0.1
lora_r = 32
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

# Args
max_seq_length = 512
output_dir = ft_model_folder
per_device_train_batch_size = 8
gradient_accumulation_steps = 2
optim = "adamw_torch"
save_steps = 10
logging_steps = 1
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 1
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
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},  # set this to True in FSDP
    report_to="wandb",
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

# handle PEFT+FSDP case
trainer.model.print_trainable_parameters()
if getattr(
    trainer.accelerator.state, "fsdp_plugin", None
):  # we are using Lora, so we dont need to shard the main model layers (no optimizer nor gradients),
    # we want specific wrappers for the modules to be trained their optimizer and gradients
    # this piece of code is checking if there is fsdp, import the plugin and a auto wrap policy for LoRA
    from peft.utils.other import fsdp_auto_wrap_policy

    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

# Train

# we need a configurator with
# accelerate config
# this machine
# multi-GPU
# How many different machines will you use (use more than 1 for multi-node training)? [1]: [nodes and not GPUs] 1
# Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. NO
# Do you wish to optimize your script with torch dynamo?[yes/NO]: NO
# Do you want to use DeepSpeed? [yes/NO]: NO
# Do you want to use FullyShardedDataParallel? [yes/NO]: yes
# the type of sharding could be full model, the optimezer, the gradients or an hybrid, we want FULL_SHARD
# Do you want to offload parameters and gradients to CPU? [yes/NO]: NO
# What should be your auto wrap policy? TRANSFORMER_BASED_WRAP
# Do you want to use the model's `_no_split_modules` to wrap. Only applicable for 🤗 Transformers [yes/NO]: prevent modules to be split --> YES
# What should be your FSDP's backward prefetch policy? BACKWARD_PRE
# Do you want to enable FSDP's forward prefetch policy? [yes/NO]: NO
# Do you want to enable FSDP's `use_orig_params` feature? [YES/no]: --> use the original parameters? NO we are focusing on LoRA
# Do you want to enable CPU RAM efficient model loading? Only applicable for 🤗 Transformers models. [YES/no]: YES
# How many GPU(s) should be used for distributed training? [1]:3
# Do you wish to use FP16 or BF16 (mixed precision)? bf16

# the configs are saved here: /leonardo/home/userexternal/pdipasqu/.cache/huggingface/accelerate/default_config.yaml
