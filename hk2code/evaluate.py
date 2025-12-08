# evaluate.py — batch-score helper
# Scores every JSON in a pattern (default: runs/*.json) against a fixed ref.
# Prints a small CSV to stdout so you can redirect to a file if you want.
#
# Usage:
#   python evaluate.py --ref sample_plan.json --pattern "runs/*.json" > results.csv

import argparse, json, glob, subprocess, sys, os

def run_metrics(pred: str, ref: str):
    p = subprocess.run(
        [sys.executable, "metrics.py", "--pred", pred, "--ref", ref],
        capture_output=True, text=True, check=True
    )
    return json.loads(p.stdout)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="sample_plan.json", help="reference plan JSON")
    ap.add_argument("--pattern", default="runs/*.json", help="glob pattern for predictions")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    print("file,model,seed,length_similarity,action_f1,overall_percent,pred_actions,ref_actions")

    for f in files:
        try:
            metrics = run_metrics(f, args.ref)
            # Extract model/seed from filename if you used a naming convention
            base = os.path.basename(f)
            model = "unknown"
            seed = ""
            # Heuristic: _<modeltag>_..._seed<NUM>.json
            # e.g., 20251206_1405_qwen_t1_seed11.json
            parts = base.split("_")
            for p in parts:
                if p.lower().startswith("seed"):
                    seed = p[4:].split(".")[0]
            for p in parts:
                if p.lower() in {"qwen","phi3","mistral","qwen2.5","phi-3"} or "qwen" in p.lower() or "phi" in p.lower() or "mistral" in p.lower():
                    model = p
                    break

            print(f"{base},{model},{seed},{metrics['length_similarity']},{metrics['action_f1']},{metrics['overall_percent']},{metrics['pred_actions']},{metrics['ref_actions']}")
        except Exception as e:
            print(f"{base},ERR,ERR,ERR,ERR,ERR,ERR,ERR")

if __name__ == "__main__":
    main()
