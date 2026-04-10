#!/usr/bin/env python3
"""
inference.py — Baseline inference script for SRE Incident Response environment.

Uses the OpenAI Client to make LLM calls for deciding incident response actions.
Emits structured stdout logs strictly following the [START], [STEP], and [END] format.

Required Environment Variables:
    API_BASE_URL  — The API endpoint for the LLM.
    MODEL_NAME    — The model identifier to use for inference.
    HF_TOKEN      — Your Hugging Face / API key.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from environment import SREIncidentEnv
from models import Action, SREAction

load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def log_start(task_id: str, incident_title: str, severity: str) -> None:
    print(
        f"[{_timestamp()}] [START] task={task_id} | incident={incident_title} | severity={severity}",
        flush=True,
    )


def log_step(
    step: int,
    max_steps: int,
    action_type: str,
    reward: float,
    status: str,
) -> None:
    print(
        f"[{_timestamp()}] [STEP] step={step}/{max_steps} | action={action_type} "
        f"| reward={reward:.4f} | status={status}",
        flush=True,
    )


def log_end(task_id: str, score: float, passed: bool, resolved: bool) -> None:
    print(
        f"[{_timestamp()}] [END] task={task_id} | score={score:.4f} | passed={passed} | resolved={resolved}",
        flush=True,
    )


SYSTEM_PROMPT = """You are an expert SRE (Site Reliability Engineer) responding to a production incident.

Available actions:
- LIST_ALERTS: View current monitoring alerts
- CHECK_DASHBOARD: Check service dashboards
- RUN_QUERY: Query metrics and logs
- GET_DEPLOYMENT: Get deployment information
- ROLLBACK: Rollback to previous version
- SCALE_SERVICE: Scale service replicas
- RESTART_SERVICE: Restart a service
- TOGGLE_FEATURE: Toggle feature flags
- PAGE_TEAM: Page the on-call team
- POST_UPDATE: Post status update
- RESOLVE: Mark incident as resolved
- ESCALATE: Escalate to senior engineer
- WAIT: Wait and monitor

Respond ONLY with valid JSON in this exact format:
{"action_type": "RUN_QUERY", "target_service": "api-gateway", "parameters": {}, "message": "Checking metrics"}

Strategy tips:
- First, LIST_ALERTS and RUN_QUERY to understand the incident
- Then, GET_DEPLOYMENT to check recent changes
- If recent deployment, use ROLLBACK
- If resource exhaustion, use SCALE_SERVICE or RESTART_SERVICE
- If unclear, WAIT and monitor
- RESOLVE only when confident the issue is fixed
"""


def get_llm_action(
    task_id: str,
    incident_title: str,
    severity: str,
    affected_services: List[str],
    step_number: int,
    action_history: List[Dict],
    active_alerts: List[Dict],
) -> Action:
    """
    Use the OpenAI Client to ask the LLM for the next incident response action.
    Falls back to a heuristic if the LLM call fails.
    """
    user_message = (
        f"Task: {task_id}\n"
        f"Incident: {incident_title}\n"
        f"Severity: {severity}\n"
        f"Affected services: {', '.join(affected_services)}\n"
        f"Step: {step_number}\n"
        f"Active alerts: {len(active_alerts)}\n"
        f"Previous actions: {[a['action_type'] for a in action_history[-3:]]}\n"
        f"\nDecide the next action. Respond with JSON only."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=200,
        )

        content = response.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        parsed = json.loads(content)

        return Action(
            action_type=SREAction(parsed.get("action_type", "WAIT")),
            target_service=parsed.get("target_service"),
            parameters=parsed.get("parameters", {}),
            message=parsed.get("message"),
        )

    except Exception as e:
        print(f"[{_timestamp()}] [WARNING] LLM call failed: {e}. Using fallback.", flush=True)
        return _fallback_action(action_history, active_alerts, affected_services)


def _fallback_action(
    action_history: List[Dict],
    active_alerts: List[Dict],
    affected_services: List[str],
) -> Action:
    """Simple heuristic fallback."""
    if not action_history:
        return Action(
            action_type=SREAction.LIST_ALERTS,
            target_service=None,
            parameters={},
            message="Getting alert status",
        )

    action_types = [a["action_type"] for a in action_history]

    if "RESOLVE" not in action_types and len(action_history) >= 3:
        return Action(
            action_type=SREAction.RESOLVE,
            target_service=None,
            parameters={},
            message="Attempting to resolve incident",
        )

    if "RUN_QUERY" not in action_types:
        return Action(
            action_type=SREAction.RUN_QUERY,
            target_service=affected_services[0] if affected_services else None,
            parameters={},
            message="Querying metrics",
        )

    if "RESTART_SERVICE" not in action_types:
        return Action(
            action_type=SREAction.RESTART_SERVICE,
            target_service=affected_services[0] if affected_services else None,
            parameters={},
            message="Restarting affected service",
        )

    return Action(
        action_type=SREAction.WAIT,
        target_service=None,
        parameters={"duration_seconds": 10},
        message="Monitoring",
    )


def run_task(env: SREIncidentEnv, task_id: str) -> float:
    """
    Run a single task from reset to termination.
    Returns the grader score in [0.0, 1.0].
    """
    obs = env.reset(task_id=task_id)

    log_start(
        task_id=task_id,
        incident_title=obs.incident_title,
        severity=obs.severity.value,
    )

    step_num = 0
    while True:
        step_num += 1

        action = get_llm_action(
            task_id=task_id,
            incident_title=obs.incident_title,
            severity=obs.severity.value,
            affected_services=obs.affected_services,
            step_number=step_num,
            action_history=[ar.model_dump() for ar in obs.action_history],
            active_alerts=[a.model_dump() for a in obs.active_alerts],
        )

        result = env.step(action)

        log_step(
            step=step_num,
            max_steps=obs.max_steps,
            action_type=action.action_type.value,
            reward=result.reward,
            status=obs.incident_status.value,
        )

        obs = result.observation

        if result.done:
            break

    grader_result = env.grade(task_id=task_id)

    log_end(
        task_id=task_id,
        score=grader_result.score,
        passed=grader_result.passed,
        resolved=grader_result.details.get("resolved", False),
    )

    return grader_result.score


def main() -> None:
    missing = []
    if not API_BASE_URL:
        missing.append("API_BASE_URL")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if not HF_TOKEN:
        missing.append("HF_TOKEN")

    if missing:
        print(
            f"[{_timestamp()}] [WARNING] Missing: {', '.join(missing)}. Using fallback agent.",
            flush=True,
        )

    env = SREIncidentEnv()
    task_ids = ["easy", "medium", "hard"]
    scores: dict = {}

    t0 = time.time()

    for tid in task_ids:
        scores[tid] = run_task(env, tid)

    elapsed = time.time() - t0

    mean_score = sum(scores.values()) / len(scores)
    print(
        f"[{_timestamp()}] [SUMMARY] tasks={len(scores)} | mean_score={mean_score:.4f} "
        f"| scores={scores} | duration={elapsed:.1f}s",
        flush=True,
    )

    all_passed = all(s >= 0.6 for s in scores.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
