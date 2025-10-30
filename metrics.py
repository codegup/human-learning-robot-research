# metrics.py — simple plan comparison + optional JSON Schema validation
import argparse, json, collections
from typing import Dict, Any, Tuple
from schema import validate_plan  # uses your Pydantic v2 schema

def action_signature(a: Dict[str, Any]) -> Tuple[str, str]:
    """Compact signature for comparing actions across plans."""
    t = (a.get("type") or "").strip().lower()
    g = (a.get("goal") or "").strip().lower()[:48]
    return (t, g)

def f1_from_multisets(pred_items, ref_items) -> float:
    """Multiset F1 over action signatures."""
    pred_ctr = collections.Counter(pred_items)
    ref_ctr  = collections.Counter(ref_items)
    tp = sum((pred_ctr & ref_ctr).values())
    fp = sum((pred_ctr - ref_ctr).values())
    fn = sum((ref_ctr - pred_ctr).values())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def plan_accuracy(pred: Dict[str, Any], ref: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and compute length_similarity, action_f1, overall_percent."""
    vp = validate_plan(pred)
    vr = validate_plan(ref)
    pa, ra = vp.actions, vr.actions

    if max(len(pa), len(ra)) == 0:
        len_score = 1.0
    else:
        len_score = 1.0 - abs(len(pa) - len(ra)) / max(len(pa), len(ra))

    sig_pred = [action_signature(a.model_dump()) for a in pa]
    sig_ref  = [action_signature(a.model_dump()) for a in ra]
    f1 = f1_from_multisets(sig_pred, sig_ref)

    overall = 100.0 * (0.4 * len_score + 0.6 * f1)
    return {
        "length_similarity": round(len_score * 100, 2),
        "action_f1": round(f1 * 100, 2),
        "overall_percent": round(overall, 2),
    }

def validate_with_jsonschema(plan: Dict[str, Any], schema_path: str) -> Dict[str, Any]:
    """Optional external JSON Schema validation."""
    try:
        import jsonschema
    except Exception as e:
        return {"ok": False, "error": f"jsonschema not installed: {e}"}
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    try:
        jsonschema.validate(instance=plan, schema=schema)
        return {"ok": True}
    except jsonschema.ValidationError as e:
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare a predicted plan to a reference plan.")
    ap.add_argument("--pred", type=str, required=True, help="Path to predicted plan JSON")
    ap.add_argument("--ref",  type=str, required=True, help="Path to reference (gold) plan JSON")
    ap.add_argument("--jsonschema", type=str, default=None, help="Optional path to a JSON Schema file")
    args = ap.parse_args()

    with open(args.pred, "r", encoding="utf-8") as f:
        pred = json.load(f)
    with open(args.ref, "r", encoding="utf-8") as f:
        ref  = json.load(f)

    print(json.dumps(plan_accuracy(pred, ref), indent=2))

    if args.jsonschema:
        print(json.dumps(validate_with_jsonschema(pred, args.jsonschema), indent=2))
