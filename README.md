---
title: SRE Incident Response Environment
emoji: 🚨
colorFrom: red
colorTo: orange
sdk: docker
app_port: 7860
---

# SRE Incident Response — OpenEnv

An **OpenEnv-compliant** simulation for SRE incident response where an RL agent diagnoses and resolves production incidents using 13 available actions.

---

## Features

| Feature | Description |
|---------|-------------|
| **13 SRE Actions** | LIST_ALERTS, CHECK_DASHBOARD, RUN_QUERY, GET_DEPLOYMENT, ROLLBACK, SCALE_SERVICE, RESTART_SERVICE, TOGGLE_FEATURE, PAGE_TEAM, POST_UPDATE, RESOLVE, ESCALATE, WAIT |
| **3 Difficulty Tiers** | Easy (API issues), Medium (infrastructure), Hard (complex cascades) |
| **15 Incident Templates** | 5 per difficulty level with realistic scenarios |
| **Time-based Scoring** | Faster resolution = higher score |
| **FastAPI Server** | REST + WebSocket endpoints with Swagger docs |
| **OpenEnv Compliant** | Full Pydantic v2 models, `/grader` returns `float ∈ [0,1]` |
| **HF Spaces Ready** | Dockerfile exposing port 7860 |

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Baseline Inference
```bash
python inference.py
```

Expected output:
```
[2026-04-10 12:00:00] [START] task=easy | incident=High API Latency | severity=medium
[2026-04-10 12:00:01] [STEP] step=1/20 | action=LIST_ALERTS | reward=0.0000 | status=active
...
[2026-04-10 12:00:30] [END] task=easy | score=0.7845 | passed=True | resolved=True
```

### 3. Start the API Server
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Visit [http://localhost:7860/docs](http://localhost:7860/docs) for Swagger UI.

### 4. Docker
```bash
docker build -t sre-incident-env .
docker run -p 7860:7860 sre-incident-env
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server health check |
| `GET` | `/tasks` | List available tasks |
| `GET` | `/state` | Current environment state |
| `POST` | `/reset` | Reset environment for an incident |
| `POST` | `/step` | Execute one incident response action |
| `POST` | `/grader` | Grade episode → `score ∈ [0.0, 1.0]` |
| `WS` | `/ws/stream` | Real-time step streaming |

---

## Available Actions

| Action | Description |
|--------|-------------|
| `LIST_ALERTS` | View current monitoring alerts |
| `CHECK_DASHBOARD` | Check service dashboards |
| `RUN_QUERY` | Query metrics and logs |
| `GET_DEPLOYMENT` | Get deployment information |
| `ROLLBACK` | Rollback to previous version |
| `SCALE_SERVICE` | Scale service replicas |
| `RESTART_SERVICE` | Restart a service |
| `TOGGLE_FEATURE` | Toggle feature flags |
| `PAGE_TEAM` | Page the on-call team |
| `POST_UPDATE` | Post status update |
| `RESOLVE` | Mark incident as resolved |
| `ESCALATE` | Escalate to senior engineer |
| `WAIT` | Wait and monitor |

---

## Tasks

### Easy — API/Service Issues
- **Scenarios**: API latency, memory leaks, payment timeouts
- **Max Steps**: 20
- **Time Limit**: 300s
- **Threshold**: ≥ 0.70

### Medium — Infrastructure Issues
- **Scenarios**: DB connection exhaustion, K8s pod loops, queue lag
- **Max Steps**: 15
- **Time Limit**: 240s
- **Threshold**: ≥ 0.65

### Hard — Complex Cascades
- **Scenarios**: Multi-region failover, feature flag rollouts, network partitions
- **Max Steps**: 12
- **Time Limit**: 180s
- **Threshold**: ≥ 0.60

---

## Project Structure

```
├── models.py              # Pydantic v2 models (Action, Observation, etc.)
├── environment.py         # OpenEnv core (reset / step / state / grade)
├── inference.py           # Baseline inference with [START]/[STEP]/[END]
├── server/
│   └── app.py            # FastAPI server + WebSocket
├── Dockerfile             # HF Spaces (port 7860)
├── openenv.yaml           # OpenEnv specification
├── requirements.txt
└── README.md
```

---

## License

MIT
