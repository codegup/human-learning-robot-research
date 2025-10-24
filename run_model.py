#!/usr/bin/env python3
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, TextStreamer
from pathlib import Path
import json


def extract_json_from_response(response: str) -> bool:
    start = response.find('{')
    if start == -1:
        return False  # no '{' found

    brace_count = 0
    for i, ch in enumerate(response[start:], start=start):
        if ch == '{':
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
            if brace_count == 0:
                json_str = response[start:i+1]
                try:
                    json.loads(json_str)
                    return True
                except json.JSONDecodeError:
                    return False

    return False    

def model_test():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir_path",
        required=True,
        help="The path of the directory containing the model"
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
    ).to("cuda").eval() # move the parameters to the GPU and just evaluate model

    if model.get_input_embeddings().weight.size(0) != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    # Ensure pad token is set
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        
    prompt = input("Enter your prompt: ")
    # Prepare input and generate output
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            # choose the next token with highest probability
            do_sample=False,
            streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True),
            max_new_tokens=1000,
            # max probability sum is 0.9 to allow some randomness
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if (extract_json_from_response(response) == True):
        print("Valid json!!!!!!!")


def main():
    model_test()

if __name__ == "__main__":
    main()
