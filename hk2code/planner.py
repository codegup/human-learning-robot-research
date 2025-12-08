# planner.py — motion-centric planner (HF model -> JSON)
# Enforces exact action count; filters forbidden phrases; retries with sampling.

import argparse, json, re, torch
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from schema import validate_plan

SYSTEM = (
    "You output ONLY strict JSON for a robot plan with keys: "
    "version, task_nl, actions (list), metadata. "
    "Each action: id, type (navigate|manipulate|perceive|communicate|wait), "
    "goal, pose{x,y,theta}? object{name,category}? params{} preconditions[] postconditions[]. "
    "Return JSON ONLY between <JSON> and </JSON>. "
    "Follow the user's task exactly."
)

USER_TMPL = """Task: {task}

World/Capabilities:
- Robot can move relative to its current orientation (left/right/forward/back) and rotate in place.
- Robot can wait for a duration (seconds).
- Robot can communicate (gesture/message).

Hard Constraints:
- Do NOT mention or use: kitchen, table, glass, guest, bottle, plate, fork, spoon.
- Do NOT copy any words or numbers from any example; follow the task only.
- If the task says 'Produce exactly N actions', output exactly N actions.
- Use motion params when relevant: {{"direction":"left|right|forward|back"}}, {{"duration_s":<int>}}, {{"angular_deg":<int>}}.

Output format (structure only; placeholders shown):
<JSON>
{{ "version":"1.0", "task_nl":"{task}",
  "actions":[
    {{"id":"a1","type":"...","goal":"...","params":{{...}}}}
  ],
  "metadata":{{}}
}}
</JSON>
"""

FORBIDDEN_TOKENS = {
    "kitchen","table","glass","guest","bottle","plate","fork","spoon",
    "walk left for 4 seconds","wait two seconds","stop moving briefly","wave hello"
}

def _violates_forbidden(plan_dict: Dict[str, Any]) -> bool:
    s = json.dumps(plan_dict).lower()
    return any(tok in s for tok in FORBIDDEN_TOKENS)

def _desired_count(task: str) -> Optional[int]:
    m = re.search(r"produce exactly (\d+)\s*actions", task, flags=re.I)
    return int(m.group(1)) if m else None

# ---------- JSON extraction helpers ----------
def _sanitize(s: str) -> str:
    s = s.replace("“",'"').replace("”",'"').replace("‘","'").replace("’","'").replace("`",'"')
    s = re.sub(r"//.*", "", s)
    s = re.sub(r"/\*[\s\S]*?\*/", "", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    return s.strip()

def _first_balanced_json(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    i, depth, in_str, esc = start, 0, False, False
    while i < len(text):
        ch = text[i]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        i += 1
    return None

def extract_json(text: str) -> Dict[str, Any]:
    m = re.search(r"<JSON>\s*(\{[\s\S]*?\})\s*</JSON>", text, re.I)
    if m:
        blob = m.group(1)
        try: return json.loads(blob)
        except json.JSONDecodeError: return json.loads(_sanitize(blob))
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.I)
    if m:
        blob = m.group(1)
        try: return json.loads(blob)
        except json.JSONDecodeError: return json.loads(_sanitize(blob))
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        try: return json.loads(t)
        except json.JSONDecodeError: return json.loads(_sanitize(t))
    blob = _first_balanced_json(text)
    if blob:
        try: return json.loads(blob)
        except json.JSONDecodeError: return json.loads(_sanitize(blob))
    raise ValueError("Could not parse JSON from model output")

# ---------- HF model helpers ----------
def _dtype():
    if torch.cuda.is_available():
        return torch.float16
    try:
        if torch.backends.mps.is_available():
            return torch.float16
    except Exception:
        pass
    return torch.float32

def load(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=_dtype(), device_map="auto",
        trust_remote_code=False, attn_implementation="eager"
    )
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok, model

def build_prompt(tok, task: str) -> str:
    user = USER_TMPL.format(task=task)
    if hasattr(tok, "apply_chat_template"):
        messages = [
            {"role":"system","content":SYSTEM},
            {"role":"user","content":user},
        ]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"[SYSTEM]\n{SYSTEM}\n[/SYSTEM]\n[USER]\n{user}\n[/USER]\n[ASSISTANT]\n"

# ---------- Core generation ----------
def generate(task: str, model_id: str, sample: bool, temperature: float, top_p: float, top_k: int, seed: Optional[int]):
    tok, model = load(model_id)
    prompt = build_prompt(tok, task)
    inputs = tok(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _gen_once() -> Dict[str, Any]:
        gen = dict(
            max_new_tokens=700,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            repetition_penalty=1.05,
        )
        if sample:
            gen.update(dict(do_sample=True, temperature=temperature, top_p=top_p))
            if top_k and top_k > 0:
                gen["top_k"] = top_k
        else:
            gen.update(dict(do_sample=False))
        with torch.inference_mode():
            out = model.generate(**inputs, **gen)
        text = tok.decode(out[0], skip_special_tokens=True)
        return extract_json(text)

    want = _desired_count(task)
    plan_json: Optional[Dict[str, Any]] = None

    for attempt in range(3):
        plan_json = _gen_once()

        if _violates_forbidden(plan_json):
            if sample:
                torch.manual_seed((seed or 0) + attempt + 1)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all((seed or 0) + attempt + 1)
            continue

        if want is not None:
            acts = plan_json.get("actions", [])
            if len(acts) != want:
                if len(acts) > want:
                    plan_json["actions"] = acts[:want]
                else:
                    if sample:
                        torch.manual_seed((seed or 0) + attempt + 11)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all((seed or 0) + attempt + 11)
                    continue

        plan = validate_plan(plan_json)
        return plan.model_dump()

    plan = validate_plan(plan_json)
    return plan.model_dump()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    plan = generate(args.task, args.model, args.sample, args.temperature, args.top_p, args.top_k, args.seed)
    print(json.dumps(plan, indent=2))
