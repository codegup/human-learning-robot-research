#!/usr/bin/env python3
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, TextStreamer
from pathlib import Path
import json

import json
import re

def schema_similarity(schema_a_path, schema_b_str):
    # Load schemas
    with open(schema_a_path) as f:
        schema_a = json.load(f)
    schema_b = json.loads(schema_b_str)

    # Fuzzy key match
    def names_match(a, b):
        a_clean = re.sub(r'[^a-z0-9]', '', a.lower())
        b_clean = re.sub(r'[^a-z0-9]', '', b.lower())
        if a_clean == b_clean:
            return True
        if a_clean in b_clean or b_clean in a_clean:
            return True
        common = len(set(a_clean) & set(b_clean))
        max_len = max(len(a_clean), len(b_clean))
        return max_len > 0 and (common / max_len) >= 0.6

    # Collect leaf/property keys, handling objects, unions, and arrays
    def collect_keys(schema, prefix=""):
        keys = set()

        # If this node has properties, collect and descend
        props = schema.get("properties", {})
        for k, v in props.items():
            full_key = f"{prefix}.{k}" if prefix else k
            keys.add(full_key)

            # descend into objects or any node that itself has properties
            if v.get("type") == "object" or "properties" in v:
                keys |= collect_keys(v, prefix=full_key)

            # handle union types: oneOf / anyOf / allOf
            for union_key in ("oneOf", "anyOf", "allOf"):
                if union_key in v and isinstance(v[union_key], list):
                    for option in v[union_key]:
                        if option.get("type") == "object" or "properties" in option:
                            keys |= collect_keys(option, prefix=full_key)

        # handle if the current node itself is a union (rare but valid)
        for union_key in ("oneOf", "anyOf", "allOf"):
            if union_key in schema and isinstance(schema[union_key], list):
                for option in schema[union_key]:
                    if option.get("type") == "object" or "properties" in option:
                        keys |= collect_keys(option, prefix=prefix)

        # Arrays: descend into items if they are object-like
        items = schema.get("items")
        if isinstance(items, dict):
            if items.get("type") == "object" or "properties" in items:
                keys |= collect_keys(items, prefix=prefix)
        elif isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and (it.get("type") == "object" or "properties" in it):
                    keys |= collect_keys(it, prefix=prefix)

        return keys

    # Flatten both schema property names
    keys_a = collect_keys(schema_a)
    keys_b = collect_keys(schema_b)

    # Match keys from A to B (don’t reuse a B key for multiple A keys)
    matched_a = set()
    matched_b = set()
    used_b = set()
    for ka in keys_a:
        leaf_a = ka.split('.')[-1]
        for kb in keys_b:
            if kb in used_b:
                continue
            leaf_b = kb.split('.')[-1]
            if names_match(leaf_a, leaf_b):
                matched_a.add(ka)
                matched_b.add(kb)
                used_b.add(kb)
                break

    # Compute precision / recall / f1
    tp = len(matched_a)
    fp = len(keys_b - matched_b)
    fn = len(keys_a - matched_a)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Optional overlap-style similarity (intersection over union of keys)
    iou = len(matched_a) / len(keys_a | keys_b) if (keys_a | keys_b) else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "similarity_percent": round(iou * 100, 2),
        "total_keys_a": len(keys_a),
        "total_keys_b": len(keys_b),
        "matched_keys": len(matched_a)
    }

def extract_json_from_response(response):
    start = response.find('{')
    if start == -1:
        return None  # no JSON object found

    brace_count = 0
    for i, ch in enumerate(response[start:], start=start):
        if ch == '{':
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
            if brace_count == 0:
                json_str = response[start:i + 1]
                try:
                    json.loads(json_str)  # validate JSON
                    return json_str
                except json.JSONDecodeError:
                    return None

    return None

def model_test():
    parser = argparse.ArgumentParser(description="Run inference with a pretrained LLM.")
    parser.add_argument(
        "--model_dir_path",
        required=True,
        help="The path of the directory containing the model"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )

    parser.add_argument(
        "--min_new_tokens",
        type=int,
        help="Minimum number of new tokens to generate"
    )

    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=False,
        help="Enable sampling for creative output"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature; lower = more deterministic"
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling threshold"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        help="Top-k sampling cutoff"
    )

    parser.add_argument(
        "--repetition_penalty",
        type=float,
        help="Penalty for repeating tokens"
    )

    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        help="Prevent repeating n-grams of this size (0 disables)"
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        help="Number of beams for beam search (1 disables beam search)"
    )

    parser.add_argument(
        "--length_penalty",
        type=float,
        help="Adjusts preference for shorter or longer sequences"
    )

    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Stop beam search early when all beams reach EOS"
    )
    
    parser.add_argument(
        "--truth_schema_path",
        type=str,
        
        help="The path to the truth schema"
    )

    args = parser.parse_args()
    set_seed(2024)

    # Ensure CUDA is available
    assert torch.cuda.is_available(), "CUDA not available on this node"

    model_dir = args.model_dir_path

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
    ).to("cuda").eval()

    # Ensure embeddings match
    if model.get_input_embeddings().weight.size(0) != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    # Ensure pad token is set
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = input("Enter your prompt: ")
    # Tokenize and move to GPU
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")


    with torch.inference_mode():
        streamer = TextStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        outputs = model.generate(
            **inputs,
            do_sample=args.do_sample,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            early_stopping=args.early_stopping,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    llm_schema = extract_json_from_response(response)
    
    if llm_schema:
        similarity = schema_similarity(args.truth_schema_path, llm_schema)
        print(f"Schema similarity: {similarity}%")
    else:
        print("Invalid response from LLM.")



def main():
    model_test()

if __name__ == "__main__":
    main()
