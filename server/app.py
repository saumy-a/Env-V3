"""
server/app.py — FastAPI server exposing the SRE Incident Response environment.

Endpoints
---------
GET  /health      → ServerStatus
GET  /tasks       → list[TaskConfig]
GET  /state       → Observation
POST /reset       → ResetResponse
POST /step        → StepResult
POST /grader      → GraderResult   (score ∈ [0.0, 1.0])
WS   /ws/stream   → real-time step streaming
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from environment import SREIncidentEnv
from models import (
    Action,
    GradeRequest,
    GraderResult,
    Observation,
    ResetRequest,
    ResetResponse,
    ServerStatus,
    StepRequest,
    StepResult,
    TaskConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("server")

app = FastAPI(
    title="SRE Incident Response OpenEnv",
    description="OpenEnv-compliant server for SRE incident response simulation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = SREIncidentEnv()
_start_time = time.time()


@app.get("/health", response_model=ServerStatus)
async def health() -> ServerStatus:
    """Server health-check."""
    return ServerStatus(
        status="healthy",
        version="1.0.0",
        environment_ready=env.is_ready,
        current_task=(env.get_current_task().task_id if env.get_current_task() else None),
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.get("/tasks", response_model=list[TaskConfig])
async def list_tasks() -> list[TaskConfig]:
    """Return all available tasks."""
    return env.get_tasks()


@app.get("/state", response_model=Observation)
async def get_state() -> Observation:
    """Return current environment observation."""
    try:
        return env.get_state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/reset", response_model=ResetResponse)
async def reset_env(req: ResetRequest) -> ResetResponse:
    """Reset the environment and start a new incident."""
    try:
        obs = env.reset(task_id=req.task_id, seed=req.seed)
        task = env.get_current_task()
        return ResetResponse(observation=obs, task_config=task)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step", response_model=StepResult)
async def step_env(req: StepRequest) -> StepResult:
    """Execute one incident response step."""
    try:
        return env.step(req.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/grader", response_model=GraderResult)
async def grade_episode(req: GradeRequest) -> GraderResult:
    """
    Grade the current incident response episode.
    **Returns a GraderResult with `score` ∈ [0.0, 1.0].**
    """
    result = env.grade(task_id=req.task_id)
    logger.info(
        "Graded task=%s  score=%.4f  passed=%s",
        result.task_id,
        result.score,
        result.passed,
    )
    return result


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    """Stream environment steps over WebSocket."""
    await ws.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            data = await ws.receive_json()

            if data.get("type") == "reset":
                task_id = data.get("task_id", "easy")
                obs = env.reset(task_id=task_id)
                await ws.send_json({"type": "reset_ack", "observation": obs.model_dump()})
                continue

            action = Action(**data.get("action", data))
            result = env.step(action)
            await ws.send_json({"type": "step_result", **result.model_dump()})

            if result.done:
                grader_result = env.grade()
                await ws.send_json({"type": "grade_result", **grader_result.model_dump()})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
        await ws.close(code=1011, reason=str(exc))


def main() -> None:
    """Launch the OpenEnv server via uvicorn."""
    import uvicorn

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )


if __name__ == "__main__":
    main()
