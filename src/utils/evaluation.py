# Evaluate HuggingFace Trainer: https://huggingface.co/learn/nlp-course/chapter3/3#evaluate
# https://huggingface.co/docs/evaluate/a_quick_tour
from dotenv import load_dotenv
import evaluate
import os
import json
import numpy as np

load_dotenv()

FAST = os.environ.get("FAST")
hf_key = os.environ.get("HUGGING_FACE_API_KEY")


def flatten(xss):
    return [x for xs in xss for x in xs]


def split_list(lst, n):
    """
    Split a list into n parts.

    Args:
    lst (list): The list to split.
    n (int): The number of parts to split the list into.

    Returns:
    list: A list containing n sublists.
    """
    split_size = len(lst) // n
    remainder = len(lst) % n
    result = []
    start = 0
    for i in range(n):
        if i < remainder:
            end = start + split_size + 1
        else:
            end = start + split_size
        result.append(lst[start:end])
        start = end
    return result


def compute_perplexity(**kwargs):
    """
    predictions
    model
    tokenizer
    device
    """
    perplexity = evaluate.load(
        "./metrics/perplexity/perplexity.py", module_type="metric"
    )
    print("perplexity loaded")
    results = perplexity.compute(**kwargs)

    return results


def compute_rouge(**kwargs):
    """
    reference
    predictions
    """
    rouge_metric = evaluate.load("./metrics/rouge/rouge.py")
    print("rouge loaded")
    results = rouge_metric.compute(**kwargs)

    return results


def compute_accuracy(predictions, labels):
    return NotImplementedError


def compute_bleu(**kwargs):
    """
    predictions
    references
    tokenizer
    max_order
    smooth
    """
    bleu = evaluate.load("./metrics/bleu/bleu.py", module_type="metric")
    print("bleu loaded")
    results = bleu.compute(**kwargs)

    return results


def pretty_json(hp):
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))


def compute_f1():
    return NotImplementedError


# TODO computare le metriche online
def compute_metrics(eval_preds, tokenizer):

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 in the preds as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    print(decoded_preds)

    accuracy_result_cumulate = 0
    f1_result_cumulate = 0
    accuracy_metric = evaluate.load("./metrics/accuracy/accuracy.py")
    f1_metric = evaluate.load("./metrics/f1/f1.py")
    # rouge_metric = evaluate.load("./metrics/rouge/rouge.py")
    for lab, pred in zip(labels, preds):

        accuracy_result = accuracy_metric.compute(references=lab, predictions=pred)
        print("acc_", accuracy_result)

        f1_result = f1_metric.compute(references=lab, predictions=pred, average="macro")
        print("f1_", f1_result)

        accuracy_result_cumulate += accuracy_result["accuracy"]
        f1_result_cumulate += f1_result["f1"]

    # rouge_result = rouge_metric.compute(references=labels, predictions=predictions)
    # print("rouge_", rouge_result)
    return {
        "accuracy": accuracy_result_cumulate / len(preds),
        # "perplexity": perplexity_result,
        "f1": f1_result_cumulate / len(preds),
        # "rouge": rouge_result,
    }
