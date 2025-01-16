# Evaluate HuggingFace Trainer: https://huggingface.co/learn/nlp-course/chapter3/3#evaluate
# https://huggingface.co/docs/evaluate/a_quick_tour
from dotenv import load_dotenv
import os
import statistics


load_dotenv()

FAST = os.environ.get("FAST")
hf_key = os.environ.get("HUGGING_FACE_API_KEY")


def get_type_statistics(meta):
    type_statistics_dict = {}
    for m in meta:
        if m["type"] not in type_statistics_dict.keys():
            type_statistics_dict[m["type"]] = 1
        else:
            type_statistics_dict[m["type"]] += 1
    return type_statistics_dict

def get_token_statistics(meta):
    token_statistics_dict = {}
    n_token = [m["token_gpt"] for m in meta if "token_gpt" in m.keys()]
    token_statistics_dict = {
        "mean": statistics.mean(n_token),
        "max": max(n_token),
        "min": min(n_token),
        "median": statistics.median(n_token),
    }
    return token_statistics_dict

def get_language_statistics(meta):
    language_statistics_dict = {}
    for m in meta:
        if not "language" in m.keys():
            return language_statistics_dict
        if m["language"] not in language_statistics_dict.keys():
            language_statistics_dict[m["language"]] = 1
        else:
            language_statistics_dict[m["language"]] += 1
    return language_statistics_dict