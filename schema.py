# schema.py — Pydantic v2 plan schema + validator
from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ValidationError

ActionType = Literal[
    "navigate",
    "manipulate",
    "perceive",
    "communicate",
    "wait",
    "plan_patch",
]

class Pose(BaseModel):
    x: float = Field(..., description="meters")
    y: float = Field(..., description="meters")
    theta: float = Field(0.0, description="heading in radians")

class ObjectRef(BaseModel):
    name: str = Field(..., description="object name or id")
    category: Optional[str] = Field(default=None, description="semantic class")

class Action(BaseModel):
    id: str = Field(..., description="unique step id, e.g., 'a1'")
    type: ActionType
    goal: Optional[str] = Field(default=None)
    pose: Optional[Pose] = Field(default=None)
    object: Optional[ObjectRef] = Field(default=None)
    params: Dict[str, Any] = Field(default_factory=dict)
    preconditions: List[str] = Field(default_factory=list)
    postconditions: List[str] = Field(default_factory=list)

class Plan(BaseModel):
    version: str = "1.0"
    task_nl: str = Field(..., description="original natural-language task")
    actions: List[Action] = Field(..., description="ordered steps")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="misc notes/provenance")

def validate_plan(plan_json: dict) -> Plan:
    try:
        return Plan.model_validate(plan_json)  # pydantic v2
    except ValidationError as e:
        raise ValueError(f"Plan validation failed: {e}")
