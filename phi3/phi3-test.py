#!/usr/bin/env python3
import os

# turn off parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from pathlib import Path

# can be any number 
set_seed(2024)
# only run if CUDA is available
assert torch.cuda.is_available(), "CUDA not available on this node"

prompt = "Print me a very basic json schema"
script_dir = Path(__file__).parent.resolve()
target_folder = script_dir / "phi3-model"

model_checkpoint = script_dir / "phi3-model"

tokenizer = AutoTokenizer.from_pretrained(
    model_checkpoint, 
    local_files_only=True, 
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.float16,      # optional but helps memory/speed
).to("cuda").eval()                 # move model to GPU

# if needed, align embeddings when tokenizer adds special tokens
if model.get_input_embeddings().weight.size(0) != len(tokenizer):
    model.resize_token_embeddings(len(tokenizer))

if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # inputs on GPU too

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=120,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response.strip())

