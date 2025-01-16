##### THIS STUFF IS PRETTY OLD #####

# TO DOWNLOAD THE MODEL LOOK AT THE README IN SECTION "DOWNLOAD MODELS"


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import PeftModel
import os
from dotenv import load_dotenv
import sys

load_dotenv()

FAST = os.environ.get("FAST")
hf_key = os.environ.get("HUGGING_FACE_API_KEY")


def load_model(
    token: str = "",
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    folder_name: str = "llama",
):
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        use_auth_token=token,
        cache_dir=os.path.join(FAST, ".cache"),
    )
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_name,
        device_map="auto",
        use_auth_token=token,
        cache_dir=os.path.join(FAST, ".cache"),
    )

    model.save_pretrained(os.path.join(FAST, "models", folder_name))
    tokenizer.save_pretrained(os.path.join(FAST, "models", folder_name))
    return True


def merge_adapter(base_model_id: str, adapters_id: str):

    base_model_path = os.path.join(FAST, "models", base_model_id)
    adapter_model_path = os.path.join(FAST, "FTmodels", adapters_id)
    # base_model_path = os.path.join(FAST, "models", "llama")
    # adapter_model_path = os.path.join(FAST, "FTmodels", "llama_May_14_2024_11_59_dataset_002_admiring_booth")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        cache_dir=os.path.join(FAST, ".cache"),
    )
    model = PeftModel.from_pretrained(
        model,
        adapter_model_path,
        device_map="auto",
        cache_dir=os.path.join(FAST, ".cache"),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        cache_dir=os.path.join(FAST, ".cache"),
    )

    model = model.merge_and_unload()
    model.save_pretrained(os.path.join(FAST, "FTmodels_merged", adapters_id))
    tokenizer.save_pretrained(os.path.join(FAST, "FTmodels_merged", adapters_id))
    return True


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge_adapter(base_model_id=sys.argv[2], adapters_id=sys.argv[3])
    else:
        load_model(token=sys.argv[2], model_name=sys.argv[3], folder_name=sys.argv[4])
