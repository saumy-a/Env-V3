#!/usr/bin/env python3
"""
inference.py — Baseline inference script for SRE Incident Response environment.

Uses the OpenAI Client to make LLM calls for deciding incident response actions.
Emits structured stdout logs strictly following the [START], [STEP], and [END] format.

Required Environment Variables:
    API_BASE_URL  — The API endpoint for the LLM (default: https://router.huggingface.co/v1)
    MODEL_NAME    — The model identifier to use (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN      — Your Hugging Face / API key.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

from environment import SREIncidentEnv
from models import Action, SREAction

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
BENCHMARK = "sre_incident_env"
MAX_STEPS = 12
TEMPERATURE = 0.3
MAX_TOKENS = 150

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert SRE (Site Reliability Engineer) responding to a production incident.

    Available actions (respond with the action_type string only):
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

    Strategy tips:
    - First, LIST_ALERTS and RUN_QUERY to understand the incident
    - Then, GET_DEPLOYMENT to check recent changes
    - If recent deployment, use ROLLBACK
    - If resource exhaustion, use SCALE_SERVICE or RESTART_SERVICE
    - RESOLVE only when confident the issue is fixed
    - If unsure, WAIT and monitor
""").strip()


def build_user_prompt(
    task_id: str,
    incident_title: str,
    severity: str,
    affected_services: List[str],
    step: int,
    action_history: List[str],
    active_alerts_count: int,
) -> str:
    history_block = "\n".join(action_history[-4:]) if action_history else "None"
    return textwrap.dedent(f"""
        Task: {task_id}
        Incident: {incident_title}
        Severity: {severity}
        Affected services: {", ".join(affected_services)}
        Step: {step}
        Active alerts: {active_alerts_count}
        Previous actions:
        {history_block}
        Choose your next action (just the action_type string, e.g. LIST_ALERTS).
    """).strip()


def get_model_action(
    task_id: str,
    incident_title: str,
    severity: str,
    affected_services: List[str],
    step: int,
    action_history: List[str],
    active_alerts_count: int,
) -> Action:
    """Get action from LLM."""
    user_prompt = build_user_prompt(
        task_id, incident_title, severity, affected_services, step, action_history, active_alerts_count
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip().upper()

        valid_actions = [a.value for a in SREAction]
        action_type = text if text in valid_actions else "WAIT"

        return Action(
            action_type=SREAction(action_type),
            target_service=affected_services[0] if affected_services else None,
            parameters={},
            message=f"Action: {action_type}",
        )
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return _fallback_action(action_history, affected_services)


def _fallback_action(
    action_history: List[str],
    affected_services: List[str],
) -> Action:
    """Simple heuristic fallback."""
    if not action_history:
        return Action(
            action_type=SREAction.LIST_ALERTS,
            target_service=affected_services[0] if affected_services else None,
            parameters={},
            message="Getting alert status",
        )

    if len(action_history) >= 3 and "RESOLVE" not in action_history:
        return Action(
            action_type=SREAction.RESOLVE,
            target_service=None,
            parameters={},
            message="Attempting to resolve",
        )

    if "RUN_QUERY" not in action_history:
        return Action(
            action_type=SREAction.RUN_QUERY,
            target_service=affected_services[0] if affected_services else None,
            parameters={},
            message="Querying metrics",
        )

    if "RESTART_SERVICE" not in action_history:
        return Action(
            action_type=SREAction.RESTART_SERVICE,
            target_service=affected_services[0] if affected_services else None,
            parameters={},
            message="Restarting service",
        )

    return Action(
        action_type=SREAction.WAIT,
        target_service=None,
        parameters={"duration_seconds": 10},
        message="Monitoring",
    )


async def main() -> None:
    env = SREIncidentEnv()

    task_ids = ["easy", "medium", "hard"]

    for task_id in task_ids:
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False
        error = None

        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        try:
            obs = env.reset(task_id=task_id)
            action_history: List[str] = []

            for step in range(1, MAX_STEPS + 1):
                action = get_model_action(
                    task_id=task_id,
                    incident_title=obs.incident_title,
                    severity=obs.severity.value,
                    affected_services=obs.affected_services,
                    step=step,
                    action_history=action_history,
                    active_alerts_count=len(obs.active_alerts),
                )

                action_str = f"{action.action_type.value}({action.target_service or ''})"
                result = env.step(action)
                obs = result.observation

                reward = result.reward or 0.0
                done = result.done

                rewards.append(reward)
                steps_taken = step
                action_history.append(action.action_type.value)

                log_step(
                    step=step,
                    action=action_str,
                    reward=reward,
                    done=done,
                    error=error,
                )

                if done:
                    break

            grader_result = env.grade(task_id=task_id)
            score = grader_result.score
            success = grader_result.passed

        except Exception as exc:
            error = str(exc)
            print(f"[DEBUG] Episode error: {exc}", flush=True)

        finally:
            try:
                env.reset(task_id=task_id)
            except Exception:
                pass

            log_end(
                success=success,
                steps=steps_taken,
                score=score,
                rewards=rewards,
            )


if __name__ == "__main__":
    asyncio.run(main())
