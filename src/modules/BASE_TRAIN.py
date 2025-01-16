# pretty much the whole code is here:
# https://github.com/huggingface/alignment-handbook/tree/main/scripts
# some more explanation on the parameters:
# https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/fine-tune-llms-in-2024-with-trl.ipynb

from typing import Dict
from datetime import datetime
import os

import src.utils.fine_tuning as ft_utils
import src.utils.dataset as ds_utils

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard.writer import SummaryWriter
from peft import LoraConfig
from datasets import DatasetDict
import torch
from transformers import (
    PreTrainedTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

FAST = os.environ.get("FAST")


class Train:
    # https://huggingface.co/docs/transformers/perf_train_gpu_one
    def __init__(
        self,
        **kwargs,
    ) -> None:

        self.base_configs: Dict = kwargs["base_configs"]
        self.time_signature = datetime.now().strftime("%b_%d_%Y_%H_%M")
        dataset_name: str = kwargs["dataset_configs"]["dataset_name"]
        self.base_model_folder: str = os.path.join(
            FAST, "models", self.base_configs["base_model_id"]
        )
        if not os.path.exists(self.base_model_folder):
            raise ValueError(f"no model in path: {self.base_model_folder}")
        self.ft_model_id: Dict = self.base_configs["ft_model_id"]
        self.computed_ft_model_id: str = (
            f"{self.ft_model_id}_{self.time_signature}_{dataset_name}"
        )
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

        self.model_init_kwargs: Dict = {
            "quantization_config": self.bitsandbytesconfig,
            **kwargs["model_init_kwargs"]["other"],
            **dict(device_map="auto"),
        }
        self.trainigargouments: TrainingArguments = TrainingArguments(
            **kwargs["training_argouments"],
            **dict(
                output_dir=self.ft_model_folder,
            ),  # here the logging_dir is set with tensorboard callbacks
        )
        self.writer: SummaryWriter = SummaryWriter(
            self.trainigargouments.logging_dir,
        )
        self.datacollator: DataCollatorForCompletionOnlyLM = (
            (
                DataCollatorForCompletionOnlyLM(
                    **kwargs["data_collator"],
                    **dict(
                        tokenizer=self.tokenizer,
                    ),
                )
            )
            if kwargs["data_collator"]["do_data_collator"]
            else None
        )
        self.sft_trainer: Dict = kwargs["sft_trainer"]

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
            preprocess_logits_for_metrics=ft_utils.preprocess_logits_for_metrics,
            formatting_func=ft_utils.formatting_function,
            data_collator=self.datacollator,  # without the collator we are calculating the loss on both instructions and completions
            callbacks=[TensorBoardCallback(tb_writer=self.writer)],
        )

    def compute_online_metrics():
        pass

    def train(self):
        trainer = self.trainer
        print("model on: ", trainer.model.device)
        print("################## trainig STARTED ###################")
        print("### open TensorBoard on $FAST to see progress ###")
        trainer.train()
        print("################## trainig FINISHED ##################")
        trainer.save_model(self.ft_model_folder)
