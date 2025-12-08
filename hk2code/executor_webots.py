# executor_webots.py — minimal stub executor to complete the pipeline
# In a real Webots integration, this would connect to the simulator and run.
# For now, it just echoes each action as successfully "executed" so you can
# validate end-to-end flow.

from typing import Dict, Any, List

def execute_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    history: List[Dict[str, Any]] = []
    for a in plan.get("actions", []):
        history.append({
            "id": a.get("id"),
            "type": a.get("type"),
            "goal": a.get("goal"),
            "status": "ok"
        })
    return {
        "success": True,
        "history": history
    }

if __name__ == "__main__":
    # tiny manual test:
    sample = {
        "version":"1.0",
        "task_nl":"demo",
        "actions":[
            {"id":"a1","type":"navigate","goal":"forward","params":{"direction":"forward","duration_s":2}},
            {"id":"a2","type":"wait","goal":"pause","params":{"duration_s":1}}
        ],
        "metadata":{}
    }
    import json
    print(json.dumps(execute_plan(sample), indent=2))
