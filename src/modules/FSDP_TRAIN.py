# pretty much the whole code is here:
# https://github.com/huggingface/alignment-handbook/tree/main/scripts
# some more explanation on the parameters:
# https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/fine-tune-llms-in-2024-with-trl.ipynb

from typing import Dict
from datetime import datetime
import os
import funkybob

import src.utils.fine_tuning as ft_utils
import src.utils.dataset as ds_utils
import src.utils.evaluation as ev_utils

from torch.utils.tensorboard import SummaryWriter
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from torch.utils.tensorboard.writer import SummaryWriter
from peft import LoraConfig
from datasets import DatasetDict
import torch
from transformers import (
    PreTrainedTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from accelerate import PartialState

FAST = os.environ.get("FAST")


class Train:
    # https://huggingface.co/docs/transformers/perf_train_gpu_one
    def __init__(
        self,
        **kwargs,
    ) -> None:

        model_names_candidates = funkybob.UniqueRandomNameGenerator(
            seed=(
                kwargs["base_configs"]["name_seed"]
                if "name_seed" in kwargs["base_configs"]
                else int(os.environ.get("SLURM_JOB_ID"))
            )
        )
        name = model_names_candidates[0]
        self.time_signature = datetime.now().strftime("%b_%d_%Y_%H_%M")
        self.base_configs: Dict = kwargs["base_configs"]
        dataset_name: str = kwargs["dataset_configs"]["dataset_name"]
        self.base_model_folder: str = os.path.join(
            FAST, "models", self.base_configs["base_model_id"]
        )

        self.ft_model_id: Dict = self.base_configs["ft_model_id"]

        checkpoint_path: str = kwargs["training_argouments"]["resume_from_checkpoint"]
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                checkpoint_number = checkpoint_path.split("/")[-1].replace("-", "")
                past_model_name = "_".join(
                    checkpoint_path.split("/")[-2].split("_")[-2:]
                )
                self.computed_ft_model_id: str = (
                    f"{self.ft_model_id}_{self.time_signature}_{dataset_name}_{past_model_name}_{checkpoint_number}_{name}"
                )
            else:
                raise ValueError(f"no checkpoint in path: {checkpoint_path}")
        else:
            self.computed_ft_model_id: str = (
                f"{self.ft_model_id}_{self.time_signature}_{dataset_name}_{name}"
            )
            kwargs["training_argouments"].pop("resume_from_checkpoint")
        print(f"the model will be saved to {self.computed_ft_model_id}")
        if not os.path.exists(self.base_model_folder):
            raise ValueError(f"no model in path: {self.base_model_folder}")

        self.ft_model_folder: str = os.path.join(
            FAST, "FTmodels", self.computed_ft_model_id
        )

        self.tokenizer: PreTrainedTokenizer = ft_utils.load_tokenizer(
            **kwargs["tokenizer_configs"],
            **dict(base_model_folder=self.base_model_folder),
        )
        if not kwargs["dataset_configs"]["test_size"]:
            raise ValueError("during Train, it is required to specify the test size")

        self.dataset: DatasetDict = ds_utils.load_custom_dataset(
            **kwargs["dataset_configs"],
            **dict(
                chat_template_func=ft_utils.apply_chat_template,
                tokenizer=self.tokenizer,
            ),
        )

        self.loraconfig: LoraConfig = LoraConfig(**kwargs["lora_config"])
        self.bitsandbytesconfig: BitsAndBytesConfig = BitsAndBytesConfig(
            **kwargs["model_init_kwargs"]["quantization"],
            **dict(
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )

        # device_string = PartialState().process_index
        device_map = "FSDP"
        self.model_init_kwargs: Dict = {
            "quantization_config": self.bitsandbytesconfig,
            **kwargs["model_init_kwargs"]["other"],
            # **dict(device_map=device_map),
        }
        self.trainigargouments: TrainingArguments = TrainingArguments(
            **kwargs["training_argouments"],
            **dict(
                output_dir=self.ft_model_folder,
                logging_dir=os.path.join(self.ft_model_folder, "runs", name),
            ),  # here the logging_dir is set with tensorboard callbacks
        )
        # self.writer: SummaryWriter = SummaryWriter(
        #     self.trainigargouments.logging_dir,
        # )
        self.logging_dir_tb = self.trainigargouments.logging_dir
        self.datacollator: DataCollatorForCompletionOnlyLM = (
            (
                DataCollatorForCompletionOnlyLM(
                    **kwargs["data_collator"]["config"],
                    **dict(
                        tokenizer=self.tokenizer,
                    ),
                )
            )
            if kwargs["data_collator"]["do_data_collator"]
            else None
        )
        if not self.datacollator:
            print("no datacollator passed")
        self.sft_trainer: Dict = kwargs["sft_trainer"]
        self.train_args = kwargs

    @property
    def trainer(self):
        """
        initialize a qLoRa trainer
        """

        dataset = self.dataset
        tokenizer = self.tokenizer
        return SFTTrainer(
            model=self.base_model_folder,
            model_init_kwargs=self.model_init_kwargs,
            args=self.trainigargouments,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            dataset_text_field=self.sft_trainer["dataset_text_field"],
            tokenizer=tokenizer,  # used by the trainer class to turn the messages converted by the chat template in input ids
            packing=self.sft_trainer["packing"],
            peft_config=self.loraconfig,
            max_seq_length=tokenizer.model_max_length,
            # preprocess_logits_for_metrics=ft_utils.preprocess_logits_for_metrics,
            # formatting_func=ft_utils.formatting_function,
            data_collator=self.datacollator,  # without the collator we are calculating the loss on both instructions and completions
            # callbacks=[TensorBoardCallback(tb_writer=self.writer)],
        )

    def compute_online_metrics():
        pass

    def train(self):
        trainer = self.trainer
        model_size = ft_utils.get_model_size(trainer.model)
        print(f"model on: {trainer.model.device}, {model_size}")
        free_memory, total_memory = [i / 1024**3 for i in torch.cuda.mem_get_info()]
        print(f"free_memory: {free_memory}, total_memory: {total_memory}")
        if PartialState().is_main_process:
            print("################## trainig STARTED ###################")
            print(
                f"### open TensorBoard on $FAST to see progress in {self.ft_model_folder} ###"
            )
            writer = SummaryWriter(
                os.path.join(
                    self.ft_model_folder,
                    "runs",
                    self.logging_dir_tb,
                )
            )
            writer.add_text(
                tag=f"train_argouments",
                text_string=ev_utils.pretty_json(self.train_args),
                global_step=0,
            )
        if getattr(
            trainer.accelerator.state, "fsdp_plugin", None
        ):  # we are using Lora, so we dont need to shard the main model layers (no optimizer nor gradients),
            # we want specific wrappers for the modules to be trained their optimizer and gradients
            # this piece of code is checking if there is fsdp, import the plugin and a auto wrap policy for LoRA
            from peft.utils.other import fsdp_auto_wrap_policy

            fsdp_plugin = trainer.accelerator.state.fsdp_plugin
            fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

        # trainer.train(
        #     resume_from_checkpoint=self.train_args["training_argouments"].get(
        #         "resume_from_checkpoint"
        #     )
        # )
        # (
        #     print("################## trainig FINISHED ##################")
        #     if PartialState().is_main_process
        #     else None
        # )
        trainer.save_model(self.ft_model_folder)
