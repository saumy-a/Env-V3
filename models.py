"""
models.py — Pydantic v2 models for SRE Incident Response Environment.

All data structures for the incident response simulation.
"""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────
#  Enums
# ──────────────────────────────────────────────


class SREAction(str, Enum):
    """Available SRE agent actions for incident response."""

    LIST_ALERTS = "LIST_ALERTS"
    CHECK_DASHBOARD = "CHECK_DASHBOARD"
    RUN_QUERY = "RUN_QUERY"
    GET_DEPLOYMENT = "GET_DEPLOYMENT"
    ROLLBACK = "ROLLBACK"
    SCALE_SERVICE = "SCALE_SERVICE"
    RESTART_SERVICE = "RESTART_SERVICE"
    TOGGLE_FEATURE = "TOGGLE_FEATURE"
    PAGE_TEAM = "PAGE_TEAM"
    POST_UPDATE = "POST_UPDATE"
    RESOLVE = "RESOLVE"
    ESCALATE = "ESCALATE"
    WAIT = "WAIT"


class Severity(str, Enum):
    """Incident severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DifficultyLevel(str, Enum):
    """Task difficulty tiers."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class IncidentStatus(str, Enum):
    """Current status of an incident."""

    ACTIVE = "active"
    INVESTIGATING = "investigating"
    MITIGATING = "mitigating"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


# ──────────────────────────────────────────────
#  Incident Models
# ──────────────────────────────────────────────


class Alert(BaseModel):
    """Single alert from monitoring."""

    alert_id: str
    name: str
    severity: Severity
    service: str
    message: str
    timestamp: float
    is_active: bool = True


class MetricSample(BaseModel):
    """Single metric data point."""

    metric_name: str
    value: float
    unit: str
    timestamp: float


class DeploymentInfo(BaseModel):
    """Deployment/version information for a service."""

    service: str
    version: str
    image: str
    replicas: int
    status: str
    last_deployed: float


class LogEntry(BaseModel):
    """Single log line."""

    timestamp: float
    level: str
    service: str
    message: str


class ActionResult(BaseModel):
    """Result of executing an SRE action."""

    action: SREAction
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    execution_time_ms: float = 0.0


# ──────────────────────────────────────────────
#  Observation Model
# ──────────────────────────────────────────────


class Observation(BaseModel):
    """OpenEnv observation — everything the agent can see."""

    step_number: int = Field(..., ge=0)
    max_steps: int = Field(..., ge=1)
    incident_id: str
    incident_title: str
    incident_status: IncidentStatus
    severity: Severity
    affected_services: List[str]
    active_alerts: List[Alert] = Field(default_factory=list)
    recent_metrics: List[MetricSample] = Field(default_factory=list)
    recent_logs: List[LogEntry] = Field(default_factory=list)
    action_history: List[ActionResult] = Field(default_factory=list)
    time_elapsed_seconds: float = 0.0
    resolution_time_remaining: float = Field(..., description="Seconds until SLA breach")
    is_terminal: bool = False


# ──────────────────────────────────────────────
#  Action Model
# ──────────────────────────────────────────────


class Action(BaseModel):
    """OpenEnv action — the agent's decision each step."""

    action_type: SREAction = Field(..., description="SRE action to execute")
    target_service: Optional[str] = Field(None, description="Target service name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    message: Optional[str] = Field(None, description="Update/status message")


# ──────────────────────────────────────────────
#  Step Result
# ──────────────────────────────────────────────


class StepResult(BaseModel):
    """Result of a single environment step."""

    observation: Observation
    reward: float = Field(..., description="Scalar reward signal")
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)
    action_result: Optional[ActionResult] = None


# ──────────────────────────────────────────────
#  Task / Grading Models
# ──────────────────────────────────────────────


class TaskConfig(BaseModel):
    """OpenEnv task definition."""

    task_id: str
    task_name: str
    difficulty: DifficultyLevel
    description: str
    scenario_type: str
    max_steps: int = Field(default=20, ge=1)
    success_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    time_limit_seconds: float = Field(default=300.0, gt=0)
    hints: List[str] = Field(default_factory=list)


class GraderResult(BaseModel):
    """
    Result from the /grader endpoint.
    `score` is **always** a float clamped to [0.0, 1.0].
    """

    task_id: str
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score ∈ [0.0, 1.0]")
    passed: bool = Field(default=False, description="score ≥ success_threshold")
    details: Dict[str, Any] = Field(default_factory=dict)
    graded_at: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: Any) -> float:
        """Guarantee the score is a float in [0.0, 1.0]."""
        return max(0.0, min(1.0, float(v)))


# ──────────────────────────────────────────────
#  API Request / Response Models
# ──────────────────────────────────────────────


class ResetRequest(BaseModel):
    """POST /reset body."""

    task_id: Optional[str] = Field(default=None, description="Task to load (easy/medium/hard)")
    seed: Optional[int] = Field(default=None, description="RNG seed override")


class ResetResponse(BaseModel):
    """POST /reset response."""

    observation: Observation
    task_config: TaskConfig


class StepRequest(BaseModel):
    """POST /step body."""

    action: Action


class GradeRequest(BaseModel):
    """POST /grader body."""

    task_id: str
    episode_log: Optional[Dict[str, Any]] = None


class ServerStatus(BaseModel):
    """GET /health response."""

    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    environment_ready: bool = Field(default=False)
    current_task: Optional[str] = None
    uptime_seconds: float = Field(default=0.0, ge=0.0)
