# schema.py — Pydantic models + validation for robot plan JSON (Pydantic v2)
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ValidationError

class Pose(BaseModel):
    x: Optional[float] = None
    y: Optional[float] = None
    theta: Optional[float] = None

class Obj(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None

class Action(BaseModel):
    id: str = Field(..., description="Unique action id like a1, a2, ...")
    type: str = Field(..., description="navigate|manipulate|perceive|communicate|wait")
    goal: Optional[str] = None
    pose: Optional[Pose] = None
    object: Optional[Obj] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    preconditions: List[str] = Field(default_factory=list)
    postconditions: List[str] = Field(default_factory=list)

class Plan(BaseModel):
    version: str
    task_nl: str
    actions: List[Action]
    metadata: Dict[str, Any] = Field(default_factory=dict)

def validate_plan(plan_json: Dict[str, Any]) -> Plan:
    try:
        return Plan.model_validate(plan_json)
    except ValidationError as e:
        raise ValueError(f"Plan validation failed: {e}")
