import os
from typing import Optional

from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall,
    context_relevancy,
    context_entity_recall,
    answer_similarity,
)
from ragas.metrics.critique import (
    harmfulness,
    maliciousness,
    coherence,
    correctness,
    conciseness,
)
from datasets import Dataset, load_dataset
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from ragas.embeddings import HuggingfaceEmbeddings
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    pipeline,
)
from accelerate import Accelerator, PartialState
from accelerate.utils import gather_object
from torch.utils.tensorboard import SummaryWriter
import torch

FAST = os.environ.get("FAST")

class AutoEvaluate:

    def __init__(
        self,
        model_id: str = None,
        evaluator_model: str = None,
        embedding_model: str = None,
        **kwargs,
    ) -> None:
        """ """

        self.model_path = os.path.join(FAST, "FTmodels", model_id)
        start, finish = kwargs["dataset_configs"]["subset"]
        self.dataset = (
            Dataset.load_from_disk(
                os.path.join(
                    self.model_path,
                    "predictions",
                    kwargs["dataset_configs"]["eval_dataset"],
                )
            )
            .shuffle(seed=kwargs["dataset_configs"]["shuffle_seed"])
            .select(
                range(
                    start,
                    finish,
                )
            )
            .filter(lambda x: bool(x["prediction"]))
        )
        self.writer = SummaryWriter(
            os.path.join(
                self.model_path,
                "runs",
                kwargs["writer_configs"]["output_logging_folder"],
            )
        )

        self.bitsandbytesconfig: BitsAndBytesConfig = (
            BitsAndBytesConfig(
                **kwargs["model_init_kwargs"]["quantization"],
                **dict(
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
            )
            if kwargs["model_init_kwargs"]["do_quantization"] == True
            else None
        )
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            os.path.join(FAST, "models", evaluator_model),
            device_map={"":PartialState().process_index},
            #torch_dtype=torch.bfloat16,
            #cache_dir=os.path.join(FAST, ".cache"),
            #**{"quantization_config": self.bitsandbytesconfig},
            quantization_config=self.bitsandbytesconfig
        )

        #self.bitsandbytesconfig: BitsAndBytesConfig = BitsAndBytesConfig(
        #    **kwargs["model_init_kwargs"]["quantization"],
        #    **dict(
        #        bnb_4bit_compute_dtype=torch.bfloat16,
        #    ),
        #) if kwargs["model_init_kwargs"]["quantization"]["do_quantization"] == True else None
        #model = AutoModelForCausalLM.from_pretrained(
        #    os.path.join(FAST, "models", evaluator_model),
        #    device_map="auto",
        #    #device_map={"":PartialState().process_index},
        #    quantization_config=self.bitsandbytesconfig,
        #)

        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(FAST, "models", evaluator_model)
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=True,
            repetition_penalty=1.1,
            temperature=0.1,
            #do_sample=True,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
        )
        pipe.tokenizer.pad_token_id = model.config.eos_token_id
        self.client = HuggingFacePipeline(pipeline=pipe)

        self.embedding = HuggingfaceEmbeddings(
            model_name=os.path.join(FAST, "models", embedding_model),
            # model_kwargs={"device": "auto"},
        )

    def auto_evaluator(self):
        # Evaluation
        try:
            generated_data = {
                "question": [d for d in self.dataset["question"]],
                "answer": [d for d in self.dataset["prediction"]],
                "contexts": [[d] for d in self.dataset["context"]],
                "ground_truth": [d for d in self.dataset["answer"]],
            }
            generated_data_dict = Dataset.from_dict(generated_data)
            metrics=[
                # faithfulness,           #QAC  X
                # answer_relevancy,       #QAC  X
                answer_correctness,       #QAG
                # context_precision,      #QAC
                # context_recall,         #QAC  X
                # context_relevancy,      #QC   
                # context_entity_recall,  #CG   X
                answer_similarity,        #QAG
                # harmfulness,            #QAC  X
                # maliciousness,
                # coherence,    
                # correctness,  
                conciseness
            ]
            score = evaluate(
                generated_data_dict,
                metrics,
                llm=self.client,
                embeddings=self.embedding,
                run_config=RunConfig(max_retries=0, max_wait=10),
            )
            print(f"---------------------SCORE:{score}")
        except Exception as e:
            print("ERROR during EVALUATE")
            print(e)
        self.writer.flush()

            # Write metrics values to TensorFlow
        for metric, value in score.items():
            self.writer.add_scalar(
                tag="AUTOEVAL/" + metric,
                scalar_value=torch.tensor(value),
            )
        self.writer.close()


    def auto_evaluator_acc(self):
        # Evaluation
        try:
            accelerator = Accelerator()
            generated_data = {
                "question": [d for d in self.dataset["question"]],
                "answer": [d for d in self.dataset["prediction"]],
                "contexts": [[d] for d in self.dataset["context"]],
                "ground_truth": [d for d in self.dataset["answer"]],
            }
            metrics=[
                # faithfulness,           #QAC  X
                # answer_relevancy,       #QAC  X
                answer_correctness,       #QAG
                # context_precision,      #QAC
                # context_recall,         #QAC  X
                # context_relevancy,      #QC   
                # context_entity_recall,  #CG   X
                answer_similarity,        #QAG
                # harmfulness,            #QAC  X
                # maliciousness,
                coherence,    
                correctness,  
                conciseness
            ]
            accelerator.wait_for_everyone()
            
            with accelerator.split_between_processes(generated_data) as data_split:
                data_split_d = Dataset.from_dict(data_split)
                score_batch = []
                batches = [data_split_d[i : i + 5] for i in range(0, len(data_split_d), 5)]
                with torch.inference_mode():
                    for batch in batches:
                        score_dict = {}
                        batch = Dataset.from_dict(batch)
                        for metric in metrics:
                            score = evaluate(
                                batch,
                                [metric],
                                llm=self.client,
                                embeddings=self.embedding,
                                run_config=RunConfig(max_retries=0, max_wait=10),
                            )
                            print(f"---------------------SCORE:{score}")
                            score_dict.update(score)
                            print(f"---------------------SCORE_DICT:{score_dict}")
                        score_batch.append(score_dict)
                    #score_dict = [score_dict]
                    print(f"---------------------SCORE_BATCH:{score_batch}")
            
            score_gathered = gather_object(score_batch)
            if accelerator.is_main_process:
                print(f"---------------------SCORE_GATHERED:{score_gathered}")
                num_splits = len(score_gathered)
                scores = {}
                for metric, value in score_gathered[0].items():
                    scores[metric] = float("{:.4f}".format(sum([s[metric] for s in score_gathered])/num_splits))
                print(f"---------------------SCORES:{scores}")
        except Exception as e:
            print("ERROR during EVALUATE")
            print(e)
        
        if accelerator.is_main_process:
            self.writer.flush()
            # Write metrics values to TensorFlow
            for metric, value in scores.items():
                self.writer.add_scalar(
                    tag="AUTOEVAL/" + metric,
                    scalar_value=torch.tensor(value),
                )
            self.writer.close()