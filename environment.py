"""
environment.py — OpenEnv-compliant SRE Incident Response environment.

Implements the standard OpenEnv interface:
  • reset(task_id) → Observation
  • step(action)   → StepResult
  • get_state()    → Observation
  • get_tasks()    → list[TaskConfig]

The RL agent orchestrates incident response by choosing the right SRE actions
to diagnose and resolve production incidents.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Dict, List, Optional

from models import (
    Action,
    ActionResult,
    Alert,
    DeploymentInfo,
    DifficultyLevel,
    GraderResult,
    IncidentStatus,
    LogEntry,
    MetricSample,
    Observation,
    SREAction,
    Severity,
    StepResult,
    TaskConfig,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Incident Scenario Templates
# ──────────────────────────────────────────────


class IncidentTemplate:
    """Template for generating an incident scenario."""

    def __init__(
        self,
        incident_id: str,
        title: str,
        severity: Severity,
        affected_services: List[str],
        root_cause: str,
        correct_actions: List[SREAction],
        alerts: List[Dict],
        metrics: List[Dict],
        logs: List[Dict],
        time_limit: float = 300.0,
    ):
        self.incident_id = incident_id
        self.title = title
        self.severity = severity
        self.affected_services = affected_services
        self.root_cause = root_cause
        self.correct_actions = correct_actions
        self.alerts = alerts
        self.metrics = metrics
        self.logs = logs
        self.time_limit = time_limit


# ──────────────────────────────────────────────
#  Incident Templates by Difficulty
# ──────────────────────────────────────────────

EASY_TEMPLATES = [
    IncidentTemplate(
        incident_id="easy_001",
        title="High API Latency on User Service",
        severity=Severity.MEDIUM,
        affected_services=["user-api", "postgres-db"],
        root_cause="Slow database queries due to missing index",
        correct_actions=[SREAction.RUN_QUERY, SREAction.RESTART_SERVICE],
        alerts=[
            {
                "alert_id": "alert_001",
                "name": "High Latency",
                "severity": Severity.MEDIUM,
                "service": "user-api",
                "message": "p99 latency > 2s",
            },
            {
                "alert_id": "alert_002",
                "name": "DB Slow Queries",
                "severity": Severity.MEDIUM,
                "service": "postgres-db",
                "message": "Slow query count > 100",
            },
        ],
        metrics=[
            {"metric_name": "api_latency_ms", "value": 2500.0, "unit": "ms"},
            {"metric_name": "db_query_ms", "value": 1500.0, "unit": "ms"},
        ],
        logs=[
            {"level": "WARN", "service": "user-api", "message": "Request timeout after 30s"},
            {"level": "ERROR", "service": "postgres-db", "message": "Slow query detected: SELECT * FROM users"},
        ],
        time_limit=300.0,
    ),
    IncidentTemplate(
        incident_id="easy_002",
        title="Memory Leak in Auth Service",
        severity=Severity.HIGH,
        affected_services=["auth-service", "redis-cache"],
        root_cause="Memory leak in session handler",
        correct_actions=[SREAction.RESTART_SERVICE, SREAction.CHECK_DASHBOARD],
        alerts=[
            {
                "alert_id": "alert_003",
                "name": "Memory Usage High",
                "severity": Severity.HIGH,
                "service": "auth-service",
                "message": "Memory > 90%",
            },
        ],
        metrics=[
            {"metric_name": "memory_percent", "value": 92.0, "unit": "%"},
            {"metric_name": "gc_pause_ms", "value": 500.0, "unit": "ms"},
        ],
        logs=[
            {"level": "WARN", "service": "auth-service", "message": "GC pause > 500ms"},
            {"level": "ERROR", "service": "auth-service", "message": "OutOfMemoryError imminent"},
        ],
        time_limit=240.0,
    ),
    IncidentTemplate(
        incident_id="easy_003",
        title="Payment Service Timeout",
        severity=Severity.HIGH,
        affected_services=["payment-service", "payment-gateway"],
        root_cause="Gateway connection pool exhaustion",
        correct_actions=[SREAction.RESTART_SERVICE, SREAction.RUN_QUERY],
        alerts=[
            {
                "alert_id": "alert_004",
                "name": "Payment Timeout",
                "severity": Severity.HIGH,
                "service": "payment-service",
                "message": "Timeout rate > 10%",
            },
        ],
        metrics=[
            {"metric_name": "timeout_rate", "value": 15.0, "unit": "%"},
            {"metric_name": "connection_pool_used", "value": 100.0, "unit": "%"},
        ],
        logs=[
            {"level": "ERROR", "service": "payment-service", "message": "Connection pool exhausted"},
        ],
        time_limit=180.0,
    ),
    IncidentTemplate(
        incident_id="easy_004",
        title="Elevated Error Rate on Checkout",
        severity=Severity.MEDIUM,
        affected_services=["checkout-service", "inventory-service"],
        root_cause="Inventory service returning 500 errors",
        correct_actions=[SREAction.RUN_QUERY, SREAction.RESTART_SERVICE],
        alerts=[
            {
                "alert_id": "alert_005",
                "name": "High Error Rate",
                "severity": Severity.MEDIUM,
                "service": "checkout-service",
                "message": "5xx rate > 5%",
            },
        ],
        metrics=[
            {"metric_name": "error_rate", "value": 8.0, "unit": "%"},
            {"metric_name": "requests_per_sec", "value": 50.0, "unit": "rps"},
        ],
        logs=[
            {"level": "ERROR", "service": "inventory-service", "message": "Database connection failed"},
        ],
        time_limit=300.0,
    ),
    IncidentTemplate(
        incident_id="easy_005",
        title="CDN Cache Miss Rate Spike",
        severity=Severity.LOW,
        affected_services=["cdn", "static-assets"],
        root_cause="Cache invalidation misconfiguration",
        correct_actions=[SREAction.CHECK_DASHBOARD, SREAction.RUN_QUERY],
        alerts=[
            {
                "alert_id": "alert_006",
                "name": "Cache Miss Rate",
                "severity": Severity.LOW,
                "service": "cdn",
                "message": "Miss rate > 30%",
            },
        ],
        metrics=[
            {"metric_name": "cache_miss_rate", "value": 45.0, "unit": "%"},
            {"metric_name": "origin_requests", "value": 10000.0, "unit": "req/s"},
        ],
        logs=[
            {"level": "WARN", "service": "cdn", "message": "Origin server overloaded"},
        ],
        time_limit=360.0,
    ),
]

MEDIUM_TEMPLATES = [
    IncidentTemplate(
        incident_id="medium_001",
        title="Database Connection Pool Exhaustion",
        severity=Severity.HIGH,
        affected_services=["api-gateway", "postgres-db"],
        root_cause="Connection leak in application code",
        correct_actions=[SREAction.RESTART_SERVICE, SREAction.RUN_QUERY, SREAction.GET_DEPLOYMENT],
        alerts=[
            {
                "alert_id": "alert_007",
                "name": "DB Connections",
                "severity": Severity.HIGH,
                "service": "postgres-db",
                "message": "Connections > 95%",
            },
            {
                "alert_id": "alert_008",
                "name": "API Errors",
                "severity": Severity.HIGH,
                "service": "api-gateway",
                "message": "5xx rate > 20%",
            },
        ],
        metrics=[
            {"metric_name": "db_connections_used", "value": 98.0, "unit": "%"},
            {"metric_name": "api_error_rate", "value": 22.0, "unit": "%"},
        ],
        logs=[
            {"level": "ERROR", "service": "api-gateway", "message": "Connection pool exhausted"},
            {"level": "WARN", "service": "postgres-db", "message": "Waiting for connections"},
        ],
        time_limit=180.0,
    ),
    IncidentTemplate(
        incident_id="medium_002",
        title="Kubernetes Pod Restarts Loop",
        severity=Severity.CRITICAL,
        affected_services=["order-service", "k8s-cluster"],
        root_cause="OOMKilled due to memory limit misconfiguration",
        correct_actions=[SREAction.GET_DEPLOYMENT, SREAction.RESTART_SERVICE, SREAction.SCALE_SERVICE],
        alerts=[
            {
                "alert_id": "alert_009",
                "name": "Pod Restarts",
                "severity": Severity.CRITICAL,
                "service": "order-service",
                "message": "Restart count > 10",
            },
            {
                "alert_id": "alert_010",
                "name": "OOMKilled",
                "severity": Severity.CRITICAL,
                "service": "order-service",
                "message": "Pod OOMKilled",
            },
        ],
        metrics=[
            {"metric_name": "pod_restart_count", "value": 15.0, "unit": "count"},
            {"metric_name": "memory_usage_mb", "value": 2048.0, "unit": "MB"},
        ],
        logs=[
            {"level": "ERROR", "service": "order-service", "message": "OOMKilled: process killed"},
        ],
        time_limit=120.0,
    ),
    IncidentTemplate(
        incident_id="medium_003",
        title="Message Queue Lag Spike",
        severity=Severity.HIGH,
        affected_services=["worker-service", "rabbitmq"],
        root_cause="Consumer group falling behind due to slow processing",
        correct_actions=[SREAction.RUN_QUERY, SREAction.SCALE_SERVICE, SREAction.RESTART_SERVICE],
        alerts=[
            {
                "alert_id": "alert_011",
                "name": "Queue Lag",
                "severity": Severity.HIGH,
                "service": "rabbitmq",
                "message": "Lag > 10000 messages",
            },
        ],
        metrics=[
            {"metric_name": "queue_lag", "value": 15000.0, "unit": "messages"},
            {"metric_name": "consumer_rate", "value": 100.0, "unit": "msg/s"},
        ],
        logs=[
            {"level": "WARN", "service": "worker-service", "message": "Processing slower than expected"},
        ],
        time_limit=240.0,
    ),
    IncidentTemplate(
        incident_id="medium_004",
        title="SSL Certificate Expiration",
        severity=Severity.HIGH,
        affected_services=["load-balancer", "api-gateway"],
        root_cause="Certificate expires in 24 hours",
        correct_actions=[SREAction.GET_DEPLOYMENT, SREAction.RUN_QUERY],
        alerts=[
            {
                "alert_id": "alert_012",
                "name": "Cert Expiring",
                "severity": Severity.HIGH,
                "service": "load-balancer",
                "message": "Certificate expires in 24h",
            },
        ],
        metrics=[
            {"metric_name": "cert_days_remaining", "value": 1.0, "unit": "days"},
            {"metric_name": "ssl_error_rate", "value": 5.0, "unit": "%"},
        ],
        logs=[
            {"level": "WARN", "service": "load-balancer", "message": "SSL certificate expiring soon"},
        ],
        time_limit=300.0,
    ),
    IncidentTemplate(
        incident_id="medium_005",
        title="Redis Cluster Failover",
        severity=Severity.HIGH,
        affected_services=["redis-cluster", "session-service"],
        root_cause="Primary node down, replica promoted",
        correct_actions=[SREAction.CHECK_DASHBOARD, SREAction.RESTART_SERVICE],
        alerts=[
            {
                "alert_id": "alert_013",
                "name": "Redis Down",
                "severity": Severity.HIGH,
                "service": "redis-cluster",
                "message": "Primary node unreachable",
            },
        ],
        metrics=[
            {"metric_name": "redis_slave_lag_ms", "value": 5000.0, "unit": "ms"},
            {"metric_name": "cache_hit_rate", "value": 60.0, "unit": "%"},
        ],
        logs=[
            {"level": "ERROR", "service": "redis-cluster", "message": "Primary node down"},
            {"level": "INFO", "service": "redis-cluster", "message": "Replica promoted to primary"},
        ],
        time_limit=180.0,
    ),
]

HARD_TEMPLATES = [
    IncidentTemplate(
        incident_id="hard_001",
        title="Multi-Region Failover Cascade",
        severity=Severity.CRITICAL,
        affected_services=["api-gateway", "us-east-1", "eu-west-1"],
        root_cause="DNS propagation delay after region failover",
        correct_actions=[SREAction.RUN_QUERY, SREAction.ROLLBACK, SREAction.ESCALATE],
        alerts=[
            {
                "alert_id": "alert_014",
                "name": "Region Down",
                "severity": Severity.CRITICAL,
                "service": "us-east-1",
                "message": "Region unavailable",
            },
            {
                "alert_id": "alert_015",
                "name": "DNS Propagation",
                "severity": Severity.HIGH,
                "service": "api-gateway",
                "message": "High latency after failover",
            },
        ],
        metrics=[
            {"metric_name": "region_availability", "value": 50.0, "unit": "%"},
            {"metric_name": "dns_ttl_remaining", "value": 30.0, "unit": "seconds"},
        ],
        logs=[
            {"level": "CRITICAL", "service": "dns", "message": "Region failover initiated"},
        ],
        time_limit=180.0,
    ),
    IncidentTemplate(
        incident_id="hard_002",
        title="Feature Flag Rollout Gone Wrong",
        severity=Severity.HIGH,
        affected_services=["recommendation-engine", "api-gateway"],
        root_cause="Feature flag causing 100% traffic to hit new code path with bug",
        correct_actions=[SREAction.TOGGLE_FEATURE, SREAction.GET_DEPLOYMENT, SREAction.RUN_QUERY],
        alerts=[
            {
                "alert_id": "alert_016",
                "name": "Error Rate",
                "severity": Severity.HIGH,
                "service": "recommendation-engine",
                "message": "Error rate > 50%",
            },
        ],
        metrics=[
            {"metric_name": "feature_flag_enabled", "value": 100.0, "unit": "%"},
            {"metric_name": "recommendation_accuracy", "value": 20.0, "unit": "%"},
        ],
        logs=[
            {"level": "ERROR", "service": "recommendation-engine", "message": "Feature flag causing errors"},
        ],
        time_limit=120.0,
    ),
    IncidentTemplate(
        incident_id="hard_003",
        title="Distributed Tracing Data Loss",
        severity=Severity.MEDIUM,
        affected_services=["tracing-collector", "jaeger", "order-service"],
        root_cause="Jaeger collector overwhelmed, spans being dropped",
        correct_actions=[SREAction.CHECK_DASHBOARD, SREAction.SCALE_SERVICE, SREAction.RUN_QUERY],
        alerts=[
            {
                "alert_id": "alert_017",
                "name": "Trace Drop Rate",
                "severity": Severity.MEDIUM,
                "service": "jaeger",
                "message": "Drop rate > 80%",
            },
        ],
        metrics=[
            {"metric_name": "span_drop_rate", "value": 85.0, "unit": "%"},
            {"metric_name": "collector_queue_size", "value": 100000.0, "unit": "spans"},
        ],
        logs=[
            {"level": "WARN", "service": "jaeger", "message": "Collector queue full, dropping spans"},
        ],
        time_limit=300.0,
    ),
    IncidentTemplate(
        incident_id="hard_004",
        title="Network Partition Between Services",
        severity=Severity.CRITICAL,
        affected_services=["user-service", "payment-service", "notification-service"],
        root_cause="istio service mesh misconfiguration blocking mTLS",
        correct_actions=[SREAction.RUN_QUERY, SREAction.GET_DEPLOYMENT, SREAction.RESTART_SERVICE],
        alerts=[
            {
                "alert_id": "alert_018",
                "name": "mTLS Errors",
                "severity": Severity.CRITICAL,
                "service": "istio",
                "message": "mTLS handshake failures",
            },
        ],
        metrics=[
            {"metric_name": "mtls_success_rate", "value": 0.0, "unit": "%"},
            {"metric_name": "inter_service_errors", "value": 1000.0, "unit": "errors/s"},
        ],
        logs=[
            {"level": "ERROR", "service": "istio", "message": "mTLS handshake failed"},
        ],
        time_limit=120.0,
    ),
    IncidentTemplate(
        incident_id="hard_005",
        title="Memory Ballooning in Java Microservices",
        severity=Severity.HIGH,
        affected_services=["catalog-service", "search-service", "recommendation-service"],
        root_cause="JDK memory leak in third-party library used by all Java services",
        correct_actions=[SREAction.RESTART_SERVICE, SREAction.CHECK_DASHBOARD, SREAction.GET_DEPLOYMENT],
        alerts=[
            {
                "alert_id": "alert_019",
                "name": "Memory Usage",
                "severity": Severity.HIGH,
                "service": "catalog-service",
                "message": "Memory > 95%",
            },
            {
                "alert_id": "alert_020",
                "name": "Memory Usage",
                "severity": Severity.HIGH,
                "service": "search-service",
                "message": "Memory > 95%",
            },
        ],
        metrics=[
            {"metric_name": "jvm_heap_used_mb", "value": 7800.0, "unit": "MB"},
            {"metric_name": "jvm_heap_max_mb", "value": 8192.0, "unit": "MB"},
        ],
        logs=[
            {"level": "WARN", "service": "catalog-service", "message": "GC overhead limit exceeded"},
        ],
        time_limit=180.0,
    ),
]


# ──────────────────────────────────────────────
#  Task Catalogue
# ──────────────────────────────────────────────

TASK_CATALOGUE: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        task_id="easy",
        task_name="SRE Incident Response - Easy",
        difficulty=DifficultyLevel.EASY,
        description="Respond to simple incidents. Use basic diagnostic and remediation actions.",
        scenario_type="api_issues",
        max_steps=20,
        success_threshold=0.7,
        time_limit_seconds=300.0,
        hints=["Check dashboards and logs first", "Restart services can clear transient issues"],
    ),
    "medium": TaskConfig(
        task_id="medium",
        task_name="SRE Incident Response - Medium",
        difficulty=DifficultyLevel.MEDIUM,
        description="Handle infrastructure and service mesh issues. Requires deeper investigation.",
        scenario_type="infrastructure",
        max_steps=15,
        success_threshold=0.65,
        time_limit_seconds=240.0,
        hints=["Query metrics to identify root cause", "Scaling may be needed for load issues"],
    ),
    "hard": TaskConfig(
        task_id="hard",
        task_name="SRE Incident Response - Hard",
        difficulty=DifficultyLevel.HARD,
        description="Complex multi-service incidents. Requires orchestrating multiple actions.",
        scenario_type="complex_cascade",
        max_steps=12,
        success_threshold=0.60,
        time_limit_seconds=180.0,
        hints=["Rollback may be faster than debugging", "Feature flags can isolate issues"],
    ),
}


# ──────────────────────────────────────────────
#  Environment Class
# ──────────────────────────────────────────────


class SREIncidentEnv:
    """
    OpenEnv-compliant environment for SRE incident response.
    """

    def __init__(self) -> None:
        self._current_task: Optional[TaskConfig] = None
        self._incident_template: Optional[IncidentTemplate] = None
        self._observation: Optional[Observation] = None
        self._step_number: int = 0
        self._episode_start: float = 0.0
        self._episode_count: int = 0
        self._is_ready: bool = False
        self._seed: int = 42
        self._action_history: List[ActionResult] = []
        self._reward_history: List[float] = []
        self._resolved: bool = False

    # ── OpenEnv interface ─────────────────────────────────────

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Observation:
        """
        Reset the environment for a new incident.
        """
        task_id = task_id or "easy"
        if task_id not in TASK_CATALOGUE:
            raise ValueError(f"Unknown task_id='{task_id}'. Available: {list(TASK_CATALOGUE)}")

        self._current_task = TASK_CATALOGUE[task_id]
        self._seed = seed if seed is not None else 42
        random.seed(self._seed)

        task = self._current_task

        if task_id == "easy":
            templates = EASY_TEMPLATES
        elif task_id == "medium":
            templates = MEDIUM_TEMPLATES
        else:
            templates = HARD_TEMPLATES

        self._incident_template = random.choice(templates)

        self._step_number = 0
        self._episode_start = time.time()
        self._episode_count += 1
        self._action_history = []
        self._resolved = False
        self._is_ready = True

        current_time = time.time()

        initial_alerts = [
            Alert(
                alert_id=a["alert_id"],
                name=a["name"],
                severity=Severity(a["severity"]),
                service=a["service"],
                message=a["message"],
                timestamp=current_time,
                is_active=True,
            )
            for a in self._incident_template.alerts
        ]

        initial_observation = Observation(
            step_number=0,
            max_steps=task.max_steps,
            incident_id=self._incident_template.incident_id,
            incident_title=self._incident_template.title,
            incident_status=IncidentStatus.ACTIVE,
            severity=self._incident_template.severity,
            affected_services=self._incident_template.affected_services,
            active_alerts=initial_alerts,
            recent_metrics=[],
            recent_logs=[],
            action_history=[],
            time_elapsed_seconds=0.0,
            resolution_time_remaining=task.time_limit_seconds,
            is_terminal=False,
        )

        self._observation = initial_observation

        logger.info(
            "Environment reset — task=%s incident=%s severity=%s",
            task_id,
            self._incident_template.incident_id,
            self._incident_template.severity,
        )

        return self._observation

    def step(self, action: Action) -> StepResult:
        """Execute one incident response step."""
        if not self._is_ready or self._current_task is None:
            raise RuntimeError("Call reset() before step()")
        if self._observation and self._observation.is_terminal:
            raise RuntimeError("Episode is done. Call reset() for a new episode.")

        self._step_number += 1
        task = self._current_task
        time_elapsed = time.time() - self._episode_start

        action_result = self._execute_action(action)
        self._action_history.append(action_result)

        if action_result.success and action.action_type in self._incident_template.correct_actions:
            reward = 1.0
        elif action_result.success:
            reward = 0.3
        else:
            reward = -0.5

        self._reward_history.append(reward)

        if action.action_type == SREAction.RESOLVE:
            if self._check_resolution(action_result):
                self._resolved = True
                reward = 10.0
            else:
                reward = -1.0
                action_result.success = False
                action_result.message = "Incident not properly resolved. Continue investigation."

        time_remaining = max(0, task.time_limit_seconds - time_elapsed)
        is_terminal = self._step_number >= task.max_steps or time_remaining <= 0 or self._resolved

        if self._resolved:
            incident_status = IncidentStatus.RESOLVED
        elif action_result.success and action.action_type in [SREAction.RESTART_SERVICE, SREAction.SCALE_SERVICE]:
            incident_status = IncidentStatus.MITIGATING
        else:
            incident_status = IncidentStatus.INVESTIGATING

        new_alerts = self._update_alerts(action, action_result)

        self._observation = Observation(
            step_number=self._step_number,
            max_steps=task.max_steps,
            incident_id=self._incident_template.incident_id,
            incident_title=self._incident_template.title,
            incident_status=incident_status,
            severity=self._incident_template.severity,
            affected_services=self._incident_template.affected_services,
            active_alerts=new_alerts,
            recent_metrics=self._observation.recent_metrics if self._observation else [],
            recent_logs=self._observation.recent_logs if self._observation else [],
            action_history=list(self._action_history),
            time_elapsed_seconds=time_elapsed,
            resolution_time_remaining=time_remaining,
            is_terminal=is_terminal,
        )

        step_result = StepResult(
            observation=self._observation,
            reward=reward,
            done=is_terminal,
            info={
                "task_id": task.task_id,
                "time_remaining": time_remaining,
            },
            action_result=action_result,
        )

        return step_result

    def _execute_action(self, action: Action) -> ActionResult:
        """Execute an SRE action and return the result."""
        action_start = time.time()
        action_type = action.action_type

        execution_times = {
            SREAction.LIST_ALERTS: 100,
            SREAction.CHECK_DASHBOARD: 200,
            SREAction.RUN_QUERY: 500,
            SREAction.GET_DEPLOYMENT: 300,
            SREAction.ROLLBACK: 2000,
            SREAction.SCALE_SERVICE: 1500,
            SREAction.RESTART_SERVICE: 1000,
            SREAction.TOGGLE_FEATURE: 500,
            SREAction.PAGE_TEAM: 300,
            SREAction.POST_UPDATE: 200,
            SREAction.RESOLVE: 100,
            SREAction.ESCALATE: 500,
            SREAction.WAIT: 1000,
        }

        execution_time = execution_times.get(action_type, 500)

        if action_type in self._incident_template.correct_actions:
            success = True
            message = f"Action {action_type.value} executed successfully."
            data = self._get_action_data(action_type, action)
        else:
            success = True
            message = f"Action {action_type.value} executed but may not address root cause."
            data = self._get_action_data(action_type, action)

        return ActionResult(
            action=action_type,
            success=success,
            message=message,
            data=data,
            execution_time_ms=float(execution_time),
        )

    def _get_action_data(self, action_type: SREAction, action: Action) -> Dict[str, Any]:
        """Get data from executing an action."""
        current_time = time.time()

        if action_type == SREAction.LIST_ALERTS:
            return {"alerts": [a.model_dump() for a in self._observation.active_alerts]}
        elif action_type == SREAction.RUN_QUERY:
            return {
                "metrics": [m.model_dump() for m in self._incident_template.metrics],
                "logs": [l.model_dump() for l in self._incident_template.logs],
            }
        elif action_type == SREAction.GET_DEPLOYMENT:
            return {
                "deployments": [
                    {
                        "service": svc,
                        "version": "v1.2.3",
                        "image": f"{svc}:latest",
                        "replicas": 3,
                        "status": "Running",
                        "last_deployed": current_time - 3600,
                    }
                    for svc in self._incident_template.affected_services
                ]
            }
        elif action_type == SREAction.CHECK_DASHBOARD:
            return {
                "dashboards": [
                    {"name": "Service Health", "status": "degraded"},
                    {"name": "Metrics Overview", "status": "warning"},
                ]
            }
        else:
            return {"status": "completed"}

    def _update_alerts(self, action: Action, result: ActionResult) -> List[Alert]:
        """Update alerts based on action taken."""
        alerts = list(self._observation.active_alerts) if self._observation else []

        if result.success and action.action_type in [SREAction.RESTART_SERVICE, SREAction.SCALE_SERVICE]:
            for alert in alerts:
                if alert.is_active and random.random() < 0.3:
                    alert.is_active = False

        if self._resolved:
            for alert in alerts:
                alert.is_active = False

        return alerts

    def _check_resolution(self, action_result: ActionResult) -> bool:
        """Check if the incident is properly resolved."""
        relevant_actions = set(self._action_history)
        required = set(self._incident_template.correct_actions[:2])

        return len(relevant_actions & required) >= 2

    def get_state(self) -> Observation:
        """Return the current observation (read-only)."""
        if self._observation is None:
            raise RuntimeError("Call reset() first")
        return self._observation

    def get_tasks(self) -> List[TaskConfig]:
        """Return all available tasks."""
        return list(TASK_CATALOGUE.values())

    def get_current_task(self) -> Optional[TaskConfig]:
        return self._current_task

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def grade(self, task_id: Optional[str] = None) -> GraderResult:
        """
        Grade the current episode.
        Returns a GraderResult with `score` ∈ [0.0, 1.0].
        """
        tid = task_id or (self._current_task.task_id if self._current_task else "easy")
        task = TASK_CATALOGUE.get(tid)
        if task is None:
            return GraderResult(
                task_id=tid,
                score=0.0,
                passed=False,
                details={"error": f"Unknown task '{tid}'"},
            )

        if not self._action_history:
            return GraderResult(
                task_id=tid,
                score=0.0,
                passed=False,
                details={"error": "No actions taken"},
            )

        resolution_score = 1.0 if self._resolved else 0.0

        time_elapsed = time.time() - self._episode_start
        time_score = max(0, 1.0 - (time_elapsed / task.time_limit_seconds))

        correct_action_count = sum(
            1 for ar in self._action_history if ar.action in self._incident_template.correct_actions
        )
        action_efficiency = min(1.0, correct_action_count / max(1, len(self._incident_template.correct_actions)))

        total_reward = sum(self._reward_history)
        reward_score = max(0, min(1.0, (total_reward + 10) / 20))

        raw_score = 0.4 * resolution_score + 0.3 * time_score + 0.2 * action_efficiency + 0.1 * reward_score

        score = max(0.0, min(1.0, raw_score))
        passed = score >= task.success_threshold

        return GraderResult(
            task_id=tid,
            score=score,
            passed=passed,
            details={
                "resolved": self._resolved,
                "time_score": round(time_score, 4),
                "action_efficiency": round(action_efficiency, 4),
                "reward_score": round(reward_score, 4),
                "total_actions": len(self._action_history),
                "correct_actions": correct_action_count,
                "success_threshold": task.success_threshold,
            },
        )
