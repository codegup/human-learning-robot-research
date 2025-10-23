#!/usr/bin/env python3
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from pathlib import Path

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def phi3_test():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir-name",
        required=True,
        help="The name of the directory containing the model"
    )
    args = parser.parse_args()

    # Reproducibility
    set_seed(2024)

    # Ensure CUDA is available
    assert torch.cuda.is_available(), "CUDA not available on this node"

    # Resolve model directory relative to script location
    model_checkpoint = Path(__file__).parent.resolve() / args.model_dir_name

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        local_files_only=True,
    ).to("cuda").eval() # move the parameters to the GPU and just evaluate model

    # Adjust embeddings if tokenizer size changed (when adding special/custom tokens)
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
            max_new_tokens=256,
            # max probability sum is 0.9 to allow some randomness
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response.strip())

def main():
    phi3_test()

if __name__ == "__main__":
    main()

