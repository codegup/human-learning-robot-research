# plannerQwen.py — Qwen-friendly planner (also works with any HF instruct model)
import argparse, json, re
from typing import Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from schema import validate_plan

#Can be replaced with other HF models, I just went with Qwen.
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM_PROMPT = """You convert natural-language tasks into a STRICT JSON plan conforming to this schema:
- keys: version, task_nl, actions (list), metadata (object)
- each action has: id, type (navigate|manipulate|perceive|communicate|wait), goal, pose {x,y,theta}?,
  object {name,category}?, params {}, preconditions [], postconditions []
Return JSON ONLY between <JSON> and </JSON>. No text outside those tags.
Ensure the actions list is non-empty.
"""

USER_TEMPLATE = """Task: <<TASK>>

World Assumptions:
- Regions: kitchen, table, door.
- Common objects: glass, bottle, plate, fork, spoon, napkin, red_block, blue_bin.
- Robot can navigate and pick/place known objects.

Constraints:
- Produce 6-10 concrete actions. Do NOT return an empty list.
- Do NOT copy the example verbatim; specialize to the task text.

Output:
<JSON>
{ "version": "1.0", "task_nl": "<<TASK>>",
  "actions": [
    { "id": "a1", "type": "navigate", "goal": "go to kitchen",
      "pose": { "x": 1.0, "y": 0.0, "theta": 0.0 }, "params": {} },

    { "id": "a2", "type": "manipulate", "goal": "pick up a glass",
      "object": { "name": "glass", "category": "container" },
      "params": { "op": "pick" } },

    { "id": "a3", "type": "navigate", "goal": "go to table",
      "pose": { "x": 3.0, "y": 0.5, "theta": 1.57 }, "params": {} },

    { "id": "a4", "type": "manipulate", "goal": "place the glass on the table",
      "object": { "name": "glass", "category": "container" },
      "params": { "op": "place" } },

    { "id": "a5", "type": "navigate", "goal": "go to guest",
      "pose": { "x": 3.2, "y": 1.2, "theta": 1.57 }, "params": {} },

    { "id": "a6", "type": "communicate", "goal": "confirm with guest",
      "params": { "message": "Water is served. Enjoy!" } }
  ],
  "metadata": {}
}
</JSON>
"""

def _pick_dtype():
    if torch.cuda.is_available():
        return torch.float16
    try:
        if torch.backends.mps.is_available():
            return torch.float16
    except Exception:
        pass
    return torch.float32

#This loads a tokenizer and model from the Hugging Face Hub
def load_model(name: str):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=_pick_dtype(),
        device_map="auto",
        trust_remote_code=False,
        attn_implementation="eager",
    )
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok, model

#This builds the chat prompt
def _build_prompt(tok, task: str) -> str:
    user = USER_TEMPLATE.replace("<<TASK>>", task)
    if hasattr(tok, "apply_chat_template"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return (
        f"<s>[SYSTEM]\n{SYSTEM_PROMPT}\n[/SYSTEM]\n"
        f"[USER]\n{user}\n[/USER]\n[ASSISTANT]\n"
    )

"""
This does its best to make the model text into a valid JSON. 

It fixes things like curly quotes, strips comments to prevent any confusion,
and any accidental trailing/duplicated commas. 
"""
def _sanitize_for_json(s: str) -> str:
    s = s.replace("“","\"").replace("”","\"").replace("‘","\"").replace("’","\"").replace("`","\"")
    s = re.sub(r'\bTrue\b', 'true', s)
    s = re.sub(r'\bFalse\b', 'false', s)
    s = re.sub(r'\bNone\b', 'null', s)
    s = re.sub(r'//.*', '', s)
    s = re.sub(r'/\*[\s\S]*?\*/', '', s)
    s = re.sub(r',(\s*[}\]])', r'\1', s)
    s = re.sub(r',\s*,+', ',', s)
    return s.strip()

"""
Scans the text and returns the first top level {...} slice with 
balanced braces. It handles strings/escapes so braces inside strings don't 
affect depth tracking. (Makes it so if there are braces in the 
task it won't get confused an end early essentially)
"""

def _balanced_brace_slice(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    i = start
    depth = 0
    in_str = False
    esc = False
    while i < len(text):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        i += 1
    return None

#This extracts a JSON object from the model output
def _extract_json(text: str) -> dict:

    #This is the Tag-wrapped JSON <JSON> </JSON>
    m = re.search(r"<JSON>\s*(\{[\s\S]*?\})\s*</JSON>", text, re.IGNORECASE)
    if m:
        blob = m.group(1)
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            return json.loads(_sanitize_for_json(blob))

    #This is the fenched code block
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if m:
        blob = m.group(1)
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            return json.loads(_sanitize_for_json(blob))

    #The raw text that is a JSON obect
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            return json.loads(_sanitize_for_json(t))

    #Lastly the first balanced-brace object that is in the text
    blob = _balanced_brace_slice(text)
    if blob:
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            return json.loads(_sanitize_for_json(blob))
    raise ValueError("Could not parse JSON plan from model output")

#Generates and validates a plan for the given tasks and retunrs a python dict
def generate_plan(task: str, model_name: str = DEFAULT_MODEL) -> Dict[str, Any]:
    tok, model = load_model(model_name)
    prompt = _build_prompt(tok, task)
    inputs = tok(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    eos_id = tok.eos_token_id
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=eos_id,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else eos_id,
        )
#Converts token IDs to text then pulls out the JSON plan
    text = tok.decode(out[0], skip_special_tokens=True)
    plan_json = _extract_json(text)

    #For string literal cleanup. Will parse again if found
    if isinstance(plan_json, str):
        plan_json = json.loads(_sanitize_for_json(plan_json))
    plan = validate_plan(plan_json)
    return plan.model_dump()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = ap.parse_args()
    plan = generate_plan(args.task, args.model)
    print(json.dumps(plan, indent=2))
