"""
Workflow Models for InternAgent

This module contains all data models and enums used throughout the workflow system,
including Ideas, Tasks, WorkflowSessions,
and WorkflowStates.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class WorkflowState(Enum):
    """Enumeration of workflow states."""
    INITIAL = "initial"
    GENERATING = "generating"
    REFLECTING = "reflecting"
    EVOLVING = "evolving"
    METHOD_DEVELOPMENT = "method_development"
    REFINING = "refining"
    RANKING = "ranking"
    AWAITING_FEEDBACK = "awaiting_feedback"
    EXTERNAL_DATA = "external_data"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class Idea:
    """Data class for research ideas (formerly hypotheses)."""
    id: str
    text: str
    score: float = 0.0
    rationale: str = ""
    baseline_summary: str = ""
    critiques: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    refine_evidence: List[Dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    scores: Dict[str, float] = field(default_factory=dict)
    references: List[Dict[str, Any]] = field(default_factory=list)
    experimental_approach: str = ""
    detailed_ideas: Dict[str, Any] = field(default_factory=dict)
    method_details: Dict[str, Any] = field(default_factory=dict)
    method_critiques: List[str] = field(default_factory=list)
    refined_method_details: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "rationale": self.rationale,
            "baseline_summary": self.baseline_summary,
            "critiques": self.critiques,
            "evidence": self.evidence,
            "refine_evidence": self.refine_evidence,
            "iteration": self.iteration,
            "scores": self.scores,
            "references": self.references,
            "experimental_approach": self.experimental_approach,
            "detailed_ideas": self.detailed_ideas,
            "method_details": self.method_details,
            "refined_method_details": self.refined_method_details,
            "method_critiques": self.method_critiques,
            "parent_id": self.parent_id,
            "generated_at": self.generated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Idea':
        """Create an Idea from a dictionary."""
        if isinstance(data.get("generated_at"), str):
            data["generated_at"] = datetime.fromisoformat(data["generated_at"])
        return cls(**data)


@dataclass
class Task:
    """Data class for research tasks (formerly research goals)."""
    id: str
    description: str
    domain: str
    constraints: List[str] = field(default_factory=list)
    background: str = ""
    ref_code_path: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "description": self.description,
            "domain": self.domain,
            "constraints": self.constraints,
            "background": self.background,
            "ref_code_path": self.ref_code_path,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create a Task from a dictionary."""
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class WorkflowSession:
    """Data class for workflow sessions."""
    id: str
    task: Task
    ideas: List[Idea] = field(default_factory=list)
    iterations_completed: int = 0
    max_iterations: int = 4
    state: WorkflowState = WorkflowState.INITIAL
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)
    top_ideas: List[str] = field(default_factory=list)
    tool_usage: Dict[str, int] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    method_phase: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "task": self.task.to_dict(),
            "ideas": [idea.to_dict() for idea in self.ideas],
            "iterations_completed": self.iterations_completed,
            "max_iterations": self.max_iterations,
            "state": self.state.value,
            "feedback_history": self.feedback_history,
            "top_ideas": self.top_ideas,
            "tool_usage": self.tool_usage,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
