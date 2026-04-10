"""
client.py — OpenEnv client for the SRE Incident Response environment.

Provides a Python client to interact with the environment server via HTTP.
"""

from __future__ import annotations

from typing import Any, List, Optional

import httpx

from models import (
    Action,
    GradeRequest,
    GraderResult,
    Observation,
    ResetRequest,
    ResetResponse,
    ServerStatus,
    StepResult,
    TaskConfig,
)


class SREIncidentClient:
    """
    HTTP client for the SRE Incident Response OpenEnv server.

    Parameters
    ----------
    base_url : str
        Server URL (default: http://localhost:7860).
    timeout : float
        Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def health(self) -> ServerStatus:
        """Check server health."""
        resp = self.client.get("/health")
        resp.raise_for_status()
        return ServerStatus(**resp.json())

    def get_tasks(self) -> List[TaskConfig]:
        """List all available tasks."""
        resp = self.client.get("/tasks")
        resp.raise_for_status()
        return [TaskConfig(**t) for t in resp.json()]

    def get_state(self) -> Observation:
        """Get current environment observation."""
        resp = self.client.get("/state")
        resp.raise_for_status()
        return Observation(**resp.json())

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> ResetResponse:
        """Reset environment for a new incident."""
        req = ResetRequest(task_id=task_id, seed=seed)
        resp = self.client.post("/reset", json=req.model_dump())
        resp.raise_for_status()
        return ResetResponse(**resp.json())

    def step(self, action: Action) -> StepResult:
        """Execute one incident response step."""
        resp = self.client.post("/step", json={"action": action.model_dump()})
        resp.raise_for_status()
        return StepResult(**resp.json())

    def grade(self, task_id: str) -> GraderResult:
        """Grade the current incident response episode."""
        req = GradeRequest(task_id=task_id)
        resp = self.client.post("/grader", json=req.model_dump())
        resp.raise_for_status()
        return GraderResult(**resp.json())

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self) -> "SREIncidentClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
