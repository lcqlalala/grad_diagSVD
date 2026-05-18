#coding:utf8
import argparse
import json
import os
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluater import ppl_eval


COMMONSENSE_DATASETS = [
    "arc_easy",
    "arc_challenge",
    "openbookqa",
    "winogrande",
    "hellaswag",
    "piqa",
    "mathqa",
]


DEFAULT_ORIGINAL_MODELS = {
    "llama-7b": "/data1/common/llm-models/llama-7b",
    "llama-2-7b": "/data1/common/llm-models/llama-2-7b",
}


def load_original_hf_model_fp16(
    model_path,
    tokenizer_path=None,
    device="cuda",
    model_seq_len=2048,
    trust_remote_code=False,
):
    """
    Load an original HuggingFace model directory in fp16.

    This is for original/uncompressed checkpoints such as llama-7b and
    llama-2-7b. It intentionally does not use torch.load/get_model_from_local,
    which are for saved .pt compressed checkpoints.
    """
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Original HF model directory not found: {model_path}")
    tokenizer_path = tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    if hasattr(model.config, "max_position_embeddings"):
        model.seqlen = int(model.config.max_position_embeddings)
    else:
        model.seqlen = int(model_seq_len)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def build_hflm(
    model=None,
    tokenizer=None,
    model_path=None,
    tokenizer_path=None,
    batch_size=1,
    device="cuda",
    trust_remote_code=False,
):
    from lm_eval.models.huggingface import HFLM

    pretrained = model_path if model_path is not None else model
    tok = tokenizer_path if tokenizer_path is not None else tokenizer
    attempts = [
        {
            "pretrained": pretrained,
            "tokenizer": tok,
            "batch_size": batch_size,
            "device": device,
            "dtype": "float16",
            "trust_remote_code": trust_remote_code,
        },
        {
            "pretrained": pretrained,
            "tokenizer": tok,
            "batch_size": batch_size,
            "device": device,
            "dtype": "float16",
        },
        {
            "pretrained": pretrained,
            "tokenizer": tok,
            "batch_size": batch_size,
            "device": device,
        },
        {
            "pretrained": pretrained,
            "tokenizer": tok,
        },
    ]
    last_error = None
    for kwargs in attempts:
        try:
            return HFLM(**kwargs)
        except TypeError as e:
            last_error = e
    raise RuntimeError(
        "Failed to initialize lm_eval.models.huggingface.HFLM. "
        "Please check your lm-eval-harness version."
    ) from last_error


def extract_main_metric(task_result):
    preferred_keys = [
        "acc_norm,none",
        "acc,none",
        "exact_match,none",
        "accuracy",
        "acc_norm",
        "acc",
        "score",
    ]
    for key in preferred_keys:
        if key in task_result:
            return key, task_result[key]
    return None, None


def evaluate_commonsense(
    model,
    tokenizer,
    eval_dataset_name,
    batch_size=1,
    device="cuda",
    num_fewshot=0,
    model_path=None,
    tokenizer_path=None,
    trust_remote_code=False,
):
    from lm_eval import evaluator

    hflm = build_hflm(
        model=model,
        tokenizer=tokenizer,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        batch_size=batch_size,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    return evaluator.simple_evaluate(
        model=hflm,
        tasks=[eval_dataset_name],
        num_fewshot=num_fewshot,
    )


def batch_evaluate_commonsense(
    model,
    tokenizer,
    batch_size=1,
    device="cuda",
    num_fewshot=0,
    model_path=None,
    tokenizer_path=None,
    trust_remote_code=False,
):
    from lm_eval import evaluator

    hflm = build_hflm(
        model=model,
        tokenizer=tokenizer,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        batch_size=batch_size,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    results = {}
    for index, dataset in enumerate(COMMONSENSE_DATASETS, 1):
        print(f"[commonsense] {index}/{len(COMMONSENSE_DATASETS)} {dataset}")
        start_time = time.time()
        try:
            raw_results = evaluator.simple_evaluate(
                model=hflm,
                tasks=[dataset],
                num_fewshot=num_fewshot,
            )
            task_result = raw_results["results"].get(dataset, {})
            metric_name, metric_value = extract_main_metric(task_result)
            results[dataset] = {
                "status": "success",
                "duration_sec": time.time() - start_time,
                "main_metric_name": metric_name,
                "main_metric_value": metric_value,
                "result": task_result,
            }
            if metric_name is not None:
                print(f"[commonsense] {dataset}: {metric_name}={metric_value:.4f}")
        except Exception as e:
            results[dataset] = {
                "status": "failed",
                "duration_sec": time.time() - start_time,
                "error": str(e),
            }
            print(f"[commonsense] {dataset} failed: {e}")

        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return results


def evaluate_mmlu(
    model,
    tokenizer,
    batch_size=1,
    device="cuda",
    shots=0,
    task_name="mmlu",
    model_path=None,
    tokenizer_path=None,
    trust_remote_code=False,
):
    from lm_eval import evaluator

    hflm = build_hflm(
        model=model,
        tokenizer=tokenizer,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        batch_size=batch_size,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    return evaluator.simple_evaluate(
        model=hflm,
        tasks=[task_name],
        num_fewshot=shots,
    )


def evaluate_original_fp16_model(
    model_path,
    tokenizer_path=None,
    model_name=None,
    eval_mode="ppl",
    dataset="wikitext2",
    model_seq_len=2048,
    eval_batch_size=1,
    device="cuda",
    commonsense_dataset="arc_easy",
    batch_commonsense=False,
    commonsense_fewshot=0,
    mmlu_shots=0,
    mmlu_task_name="mmlu",
    trust_remote_code=False,
):
    """
    Evaluate one original HF-format model in fp16.

    eval_mode:
      - ppl
      - commonsense
      - mmlu
    """
    model_name = model_name or os.path.basename(model_path.rstrip("/"))
    print("=" * 80)
    print(f"[original-fp16] model_name={model_name}")
    print(f"[original-fp16] model_path={model_path}")
    print(f"[original-fp16] eval_mode={eval_mode}")
    print(f"[original-fp16] dtype=fp16 device={device}")
    print("=" * 80)

    start_time = time.time()
    result = {
        "model_name": model_name,
        "model_path": model_path,
        "tokenizer_path": tokenizer_path or model_path,
        "dtype": "fp16",
        "device": device,
        "eval_mode": eval_mode,
        "timestamp": datetime.now().isoformat(),
    }

    if eval_mode == "ppl":
        model, tokenizer = load_original_hf_model_fp16(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            device=device,
            model_seq_len=model_seq_len,
            trust_remote_code=trust_remote_code,
        )
        ppl_eval(
            model,
            tokenizer,
            datasets=[dataset],
            model_seq_len=model_seq_len,
            batch_size=eval_batch_size,
            device=device,
        )
        result.update({
            "dataset": dataset,
            "model_seq_len": model_seq_len,
            "eval_batch_size": eval_batch_size,
        })

    elif eval_mode == "commonsense":
        if batch_commonsense:
            cs_results = batch_evaluate_commonsense(
                model=None,
                tokenizer=None,
                model_path=model_path,
                tokenizer_path=tokenizer_path or model_path,
                batch_size=eval_batch_size,
                device=device,
                num_fewshot=commonsense_fewshot,
                trust_remote_code=trust_remote_code,
            )
            result["commonsense_results"] = cs_results
        else:
            raw_results = evaluate_commonsense(
                model=None,
                tokenizer=None,
                model_path=model_path,
                tokenizer_path=tokenizer_path or model_path,
                eval_dataset_name=commonsense_dataset,
                batch_size=eval_batch_size,
                device=device,
                num_fewshot=commonsense_fewshot,
                trust_remote_code=trust_remote_code,
            )
            task_result = raw_results["results"].get(commonsense_dataset, {})
            metric_name, metric_value = extract_main_metric(task_result)
            result.update({
                "commonsense_dataset": commonsense_dataset,
                "num_fewshot": commonsense_fewshot,
                "main_metric_name": metric_name,
                "main_metric_value": metric_value,
                "result": task_result,
            })
            if metric_name is not None:
                print(f"[commonsense] {commonsense_dataset}: {metric_name}={metric_value:.4f}")

    elif eval_mode == "mmlu":
        raw_results = evaluate_mmlu(
            model=None,
            tokenizer=None,
            model_path=model_path,
            tokenizer_path=tokenizer_path or model_path,
            batch_size=eval_batch_size,
            device=device,
            shots=mmlu_shots,
            task_name=mmlu_task_name,
            trust_remote_code=trust_remote_code,
        )
        task_result = raw_results["results"].get(mmlu_task_name, {})
        metric_name, metric_value = extract_main_metric(task_result)
        result.update({
            "mmlu_task_name": mmlu_task_name,
            "mmlu_shots": mmlu_shots,
            "main_metric_name": metric_name,
            "main_metric_value": metric_value,
            "result": task_result,
        })
        if metric_name is not None:
            print(f"[mmlu] {mmlu_task_name}: {metric_name}={metric_value:.4f}")
    else:
        raise ValueError(f"Unsupported eval_mode={eval_mode}")

    result["duration_sec"] = time.time() - start_time
    return result


def evaluate_original_fp16_llama_7b_and_llama2_7b(
    llama7b_path=DEFAULT_ORIGINAL_MODELS["llama-7b"],
    llama2_7b_path=DEFAULT_ORIGINAL_MODELS["llama-2-7b"],
    eval_mode="ppl",
    dataset="wikitext2",
    model_seq_len=2048,
    eval_batch_size=1,
    device="cuda",
    save_path=None,
    **kwargs,
):
    """
    Evaluate original LLaMA-7B and LLaMA-2-7B in fp16 with the same settings.
    """
    all_results = {}
    model_specs = [
        ("llama-7b", llama7b_path),
        ("llama-2-7b", llama2_7b_path),
    ]
    for model_name, model_path in model_specs:
        result = evaluate_original_fp16_model(
            model_path=model_path,
            tokenizer_path=model_path,
            model_name=model_name,
            eval_mode=eval_mode,
            dataset=dataset,
            model_seq_len=model_seq_len,
            eval_batch_size=eval_batch_size,
            device=device,
            **kwargs,
        )
        all_results[model_name] = result
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"[original-fp16] results saved to: {save_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama7b_path", type=str, default=DEFAULT_ORIGINAL_MODELS["llama-7b"])
    parser.add_argument("--llama2_7b_path", type=str, default=DEFAULT_ORIGINAL_MODELS["llama-2-7b"])
    parser.add_argument("--eval_mode", type=str, default="ppl", choices=["ppl", "commonsense", "mmlu"])
    parser.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "ptb", "c4"])
    parser.add_argument("--model_seq_len", type=int, default=2048)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--commonsense_dataset", type=str, default="arc_easy", choices=COMMONSENSE_DATASETS)
    parser.add_argument("--batch_commonsense", action="store_true")
    parser.add_argument("--commonsense_fewshot", type=int, default=0)
    parser.add_argument("--mmlu_shots", type=int, default=0)
    parser.add_argument("--mmlu_task_name", type=str, default="mmlu")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    evaluate_original_fp16_llama_7b_and_llama2_7b(
        llama7b_path=args.llama7b_path,
        llama2_7b_path=args.llama2_7b_path,
        eval_mode=args.eval_mode,
        dataset=args.dataset,
        model_seq_len=args.model_seq_len,
        eval_batch_size=args.eval_batch_size,
        device=args.device,
        save_path=args.save_path,
        commonsense_dataset=args.commonsense_dataset,
        batch_commonsense=args.batch_commonsense,
        commonsense_fewshot=args.commonsense_fewshot,
        mmlu_shots=args.mmlu_shots,
        mmlu_task_name=args.mmlu_task_name,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
