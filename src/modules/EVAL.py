import os
from typing import Optional, Dict, List
import time
from tqdm import tqdm

import src.utils.dataset as ds_utils
import src.utils.fine_tuning as ft_utils
import src.utils.evaluation as ev_utils
import src.utils.inference as inf_utils

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    BitsAndBytesConfig,
)
from datasets import DatasetDict
from torch.utils.tensorboard import SummaryWriter
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object

# questo mi ha salvato la vita: https://medium.com/@geronimo7/llms-multi-gpu-inference-with-accelerate-5a8333e4c5db
# qui documentazione su accelerate https://huggingface.co/docs/accelerate/basic_tutorials/launch
to_mb = 1024**2

FAST = os.environ.get("FAST")


class Evaluate:

    def __init__(
        self,
        model_id: str,
        base_model: Optional[bool] = False,
        dataset: Optional[DatasetDict] = None,
        logging_file: Optional[str] = None,
        # writer: Optional[SummaryWriter] = None,
        **kwargs,
    ) -> None:
        """
        the class can be either initialized in a stand alone mode or right after a training. if it is used right after a training, in the same job,
        the model_id, dataset and writer are explicitely passed from the trainer class. in the stand alone mode, the class is initialize via the
        kwargs, that contain all the information needed to put the model on the GPU(s), create the tokenizer, the dataset and the writer.

        the stand alone mode is "active" only when the user do not pass any dataset explictely as DatasetDict. on the other hand, the Evaluate class
        requires to specify the configs in the EVALconfigs yaml.

        in stand alone mode, the folder is created inside the "run" directory with all the other runs of the specified model
        """
        self.model_id = model_id
        if base_model == True:
            self.model_path = os.path.join(FAST, "models", model_id)
        else:
            self.model_path = os.path.join(FAST, "FTmodels", model_id)
        self.interence_configs: Dict = kwargs["inference_configs"]
        if not os.path.exists(self.model_path):
            raise ValueError(f"no model in path: {self.model_path}")
        if not dataset:
            if (
                (not kwargs["dataset_configs"]["dataset_name"])
                or (not kwargs["writer_configs"]["output_logging_folder"])
                or (not kwargs["dataset_configs"]["subset"])
                or (not kwargs["dataset_configs"]["num_proc"])
                or (not model_id)
            ):
                raise ValueError(
                    "please provide split, dataset_name, output_logging_folder or num_proc to Evaluate in stand alone mode"
                )
            self.tokenizer_configs = kwargs["tokenizer_configs"]
            self.tokenizer = ft_utils.load_tokenizer(
                self.model_path, **self.tokenizer_configs
            )  # loads the tokenizer of the model we want to evaluate
            dataset = ds_utils.load_custom_dataset(
                **kwargs["dataset_configs"],
                # chat_template_func=ft_utils.apply_chat_template,  # if we are using Evaluate in stand alone mode, we want to apply the chat template to the dataset
                tokenizer=self.tokenizer,
                _eval=True,
            )
            self.dataset = dataset
            self.logging_file = kwargs["writer_configs"]["output_logging_folder"]

        else:
            self.dataset = dataset
            # self.writer = writer  # DONE il writer scrive n volte, una per ogni gpu, perchè viene instanziato nella classe... possibile risolvere, ma è inutile per ora.
            # possiamo instanziare il writer solo dentro il main process, passandogli (dalla classe) il logging file come ho fatto per la modalità stand alone
            self.logging_file = logging_file
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                cache_dir=os.path.join(FAST, ".cache"),
            )
        self.eval_args = kwargs

    def evaluator(self):
        dataset = self.dataset["test"]
        # here a comprehensive guide on accelerate launch command https://huggingface.co/docs/accelerate/concept_guides/deferring_execution
        accelerator = Accelerator()
        if accelerator.is_main_process:
            print(f"main process on {accelerator.device}")
        # with accelerator.main_process_first():  # The other processes will enter the with block after the main process exits.
        #     dataset = self.dataset["test"]
        #     questions = dataset["question"]
        #     labels = dataset["answer"]
        print(f"start processing on {accelerator.device}")

        questions = dataset["text"]
        labels = dataset["answer"]

        # memory_consumption = torch.cuda.max_memory_allocated(accelerator.device) / to_mb
        # print(
        #     f"model loading on {accelerator.device}, memory consumption before {memory_consumption} MB"
        # )
        self.bitsandbytesconfig: BitsAndBytesConfig = (
            BitsAndBytesConfig(
                **self.eval_args["model_init_kwargs"]["quantization"],
                **dict(
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
            )
            if self.eval_args["model_init_kwargs"]["do_quantization"] == True
            else None
        )

        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map={"": accelerator.process_index},
            torch_dtype=torch.bfloat16,
            cache_dir=os.path.join(FAST, ".cache"),
            **{"quantization_config": self.bitsandbytesconfig},
        )

        tokenizer = ft_utils.load_tokenizer(self.model_path, **self.tokenizer_configs)

        accelerator.wait_for_everyone()
        start = time.time()
        with accelerator.split_between_processes(questions) as prompts:
            results = dict(
                output_texts_masked=[],
                num_tokens=0,
                mask_empty_sentences=[],
                output_texts=[],
            )

            prompt_batches = inf_utils.prepare_prompts(
                prompts,
                tokenizer,
                batch_size=self.interence_configs["prepare"]["batch_size"],
            )

            for prompts_tokenized in tqdm(prompt_batches):
                output_texts_masked, num_tokens, mask_empty_sentences, output_texts = (
                    inf_utils.batch_inference(
                        tokenized_sentences=prompts_tokenized,
                        tokenizer=tokenizer,
                        model=model,
                        device=accelerator.device,
                        **self.interence_configs["predict"],
                    )
                )

                results["output_texts_masked"].extend(output_texts_masked)
                results["num_tokens"] += num_tokens
                results["mask_empty_sentences"].extend(mask_empty_sentences)
                results["output_texts"].extend(output_texts)

            len_output_texts = len(results["output_texts"])
            len_output_texts_masked = len(results["output_texts_masked"])
            print(
                f"model cuda: {model.device}, processed: {len_output_texts} out of {len_output_texts_masked}"
            )
            results = [results]

        results_gathered = gather_object(results)  # dove li gathera? sul main device?
        if accelerator.is_main_process:
            timediff = time.time() - start
            num_tokens = sum([r["num_tokens"] for r in results_gathered])
            print(
                f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}"
            )
            writer = SummaryWriter(
                os.path.join(
                    self.model_path,
                    "runs",
                    self.logging_file,
                )
            )
            writer.add_text(
                tag=f"eval_arguments",
                text_string=ev_utils.pretty_json(self.eval_args),
                global_step=0,
            )
            # print(f"main process on {accelerator.device}")
            # print(f"number of gathered results: {len(results_gathered)}")

            output_texts = ev_utils.flatten(
                [rg["output_texts"] for rg in results_gathered]
            )
            output_texts_masked = ev_utils.flatten(
                [rg["output_texts_masked"] for rg in results_gathered]
            )
            mask_empty_sentences = ev_utils.flatten(
                [rg["mask_empty_sentences"] for rg in results_gathered]
            )
            masked_labels = [l for l, m in zip(labels, mask_empty_sentences) if m == 1]
            # print(f"n predictions: {len(results)}, n labels: {len(labels)}")
            rouge_result = ev_utils.compute_rouge(
                predictions=output_texts_masked, references=masked_labels
            )
            print(rouge_result)
            perplexity_result = ev_utils.compute_perplexity(
                predictions=output_texts_masked,
                model=model,
                tokenizer=tokenizer,
                device=accelerator.device,
            )
            print(perplexity_result)
            bleu_result = ev_utils.compute_bleu(
                predictions=output_texts_masked, references=masked_labels
            )
            print(bleu_result)
            writer.add_scalar(  # TODO rouge tutto sotto un unico tag
                tag="METRICS/ROUGE1",
                scalar_value=torch.tensor(rouge_result["rouge1"]),
            )
            writer.add_scalar(
                tag="METRICS/ROUGE2",
                scalar_value=torch.tensor(rouge_result["rouge2"]),
            )
            writer.add_scalar(
                tag="METRICS/ROUGEL",
                scalar_value=torch.tensor(rouge_result["rougeL"]),
            )
            writer.add_scalar(
                tag="METRICS/ROUGELSUM",
                scalar_value=torch.tensor(rouge_result["rougeLsum"]),
            )

            writer.add_scalar(
                tag="METRICS/PERPLEXITY",
                scalar_value=torch.tensor(perplexity_result["mean_perplexity"]),
            )
            writer.add_scalar(
                tag="METRICS/BLEU",
                scalar_value=torch.tensor(bleu_result["bleu"]),
            )

            for t_pred, t_label, t_question in zip(output_texts, labels, questions):
                writer.add_text(
                    tag=f"TESTS/{t_question}",
                    text_string=t_label,
                    global_step=0,
                )
                writer.add_text(
                    tag=f"TESTS/{t_question}",
                    text_string=t_pred,
                    global_step=1,
                )
            writer.flush()
            writer.close()
            dataset: DatasetDict = dataset.add_column(
                name="prediction", column=output_texts
            )
            where = self.eval_args["dataset_configs"]["dataset_name"]
            os.makedirs(
                os.path.join(
                    self.model_path,
                    "predictions",
                    where,
                )
            )
            dataset.save_to_disk(
                dataset_path=os.path.join(
                    self.model_path,
                    "predictions",
                    where,
                )
            )

            print(
                f"dataset for model {self.model_id} saved to {where}",
            )
