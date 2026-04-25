"""
Seed SFT data for Long Horizon Memory action generation.

Each sample contains:
- observation: memory entries with explicit indices + a new incoming message
- response: action JSON matching LongHorizonMemoryAction in models.py
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


SYSTEM_PROMPT = (
    "You are a memory manager. Decide exactly one action for the new message.\n"
    "Return ONLY one JSON object with one of these forms:\n"
    '{"operation":"add"}\n'
    '{"operation":"noop"}\n'
    '{"operation":"remove","remove_index":<int>}\n'
    "Do not output markdown, explanations, or extra text."
)


SEED_DATA: List[Dict[str, Any]] = [
    {
        "observation": {
            "domain": "project_support",
            "task_name": "easy",
            "memory": [],
            "new_message": "Customer says login fails with error 401 after password reset.",
        },
        "response": {"operation": "add"},
    },
    {
        "observation": {
            "domain": "project_support",
            "task_name": "easy",
            "memory": [{"index": 0, "text": "login fails with error 401 after reset"}],
            "new_message": "Customer also mentions issue started after mobile app update v2.4.",
        },
        "response": {"operation": "add"},
    },
    {
        "observation": {
            "domain": "project_support",
            "task_name": "easy",
            "memory": [
                {"index": 0, "text": "login fails with error 401 after reset"},
                {"index": 1, "text": "issue started after app update v2.4"},
            ],
            "new_message": "I had pasta for dinner yesterday.",
        },
        "response": {"operation": "noop"},
    },
    {
        "observation": {
            "domain": "project_support",
            "task_name": "easy",
            "memory": [
                {"index": 0, "text": "login fails with error 401 after reset"},
                {"index": 1, "text": "issue started after app update v2.4"},
            ],
            "new_message": "Correction: error code is 403, not 401.",
        },
        "response": {"operation": "remove", "remove_index": 0},
    },
    {
        "observation": {
            "domain": "project_support",
            "task_name": "easy",
            "memory": [{"index": 0, "text": "issue started after app update v2.4"}],
            "new_message": "Login fails with error 403 after password reset.",
        },
        "response": {"operation": "add"},
    },
    {
        "observation": {
            "domain": "incident_response",
            "task_name": "medium",
            "memory": [],
            "new_message": "Alert: API latency p95 spiked from 210ms to 880ms.",
        },
        "response": {"operation": "add"},
    },
    {
        "observation": {
            "domain": "incident_response",
            "task_name": "medium",
            "memory": [{"index": 0, "text": "API latency p95 spiked to 880ms"}],
            "new_message": "CPU on service auth-api reached 97%.",
        },
        "response": {"operation": "add"},
    },
    {
        "observation": {
            "domain": "incident_response",
            "task_name": "medium",
            "memory": [
                {"index": 0, "text": "API latency p95 spiked to 880ms"},
                {"index": 1, "text": "auth-api CPU reached 97%"},
            ],
            "new_message": "By the way, I started learning guitar.",
        },
        "response": {"operation": "noop"},
    },
    {
        "observation": {
            "domain": "incident_response",
            "task_name": "medium",
            "memory": [
                {"index": 0, "text": "API latency p95 spiked to 880ms"},
                {"index": 1, "text": "auth-api CPU reached 97%"},
            ],
            "new_message": "Correction: CPU peak was 72%, not 97%.",
        },
        "response": {"operation": "remove", "remove_index": 1},
    },
    {
        "observation": {
            "domain": "incident_response",
            "task_name": "medium",
            "memory": [{"index": 0, "text": "API latency p95 spiked to 880ms"}],
            "new_message": "auth-api CPU peaked at 72%.",
        },
        "response": {"operation": "add"},
    },
    {
        "observation": {
            "domain": "compiler_project",
            "task_name": "hard",
            "memory": [
                {"index": 0, "text": "parser fails on nested generic types"},
                {"index": 1, "text": "need AST diff tests for semantic pass"},
                {"index": 2, "text": "memory leak in IR builder"},
                {"index": 3, "text": "deadline Friday for beta release"},
            ],
            "new_message": "My weekend hiking plan changed due to rain.",
        },
        "response": {"operation": "noop"},
    },
    {
        "observation": {
            "domain": "compiler_project",
            "task_name": "hard",
            "memory": [
                {"index": 0, "text": "parser fails on nested generic types"},
                {"index": 1, "text": "need AST diff tests for semantic pass"},
                {"index": 2, "text": "memory leak in IR builder"},
                {"index": 3, "text": "deadline Friday for beta release"},
            ],
            "new_message": "New blocker: codegen emits wrong jump target for switch-case.",
        },
        "response": {"operation": "add"},
    },
    {
        "observation": {
            "domain": "compiler_project",
            "task_name": "hard",
            "memory": [
                {"index": 0, "text": "parser fails on nested generic types"},
                {"index": 1, "text": "need AST diff tests for semantic pass"},
                {"index": 2, "text": "memory leak in IR builder"},
                {"index": 3, "text": "deadline Friday for beta release"},
                {"index": 4, "text": "wrong jump target in switch-case codegen"},
            ],
            "new_message": "Correction: leak is in optimization pass, not IR builder.",
        },
        "response": {"operation": "remove", "remove_index": 2},
    },
    {
        "observation": {
            "domain": "compiler_project",
            "task_name": "hard",
            "memory": [
                {"index": 0, "text": "parser fails on nested generic types"},
                {"index": 1, "text": "need AST diff tests for semantic pass"},
                {"index": 2, "text": "deadline Friday for beta release"},
                {"index": 3, "text": "wrong jump target in switch-case codegen"},
            ],
            "new_message": "Memory leak is in optimization pass.",
        },
        "response": {"operation": "add"},
    },
    {
        "observation": {
            "domain": "fraud_detection",
            "task_name": "medium",
            "memory": [],
            "new_message": "Chargebacks rose 18% week over week in LATAM.",
        },
        "response": {"operation": "add"},
    },
    {
        "observation": {
            "domain": "fraud_detection",
            "task_name": "medium",
            "memory": [{"index": 0, "text": "chargebacks rose 18% WoW in LATAM"}],
            "new_message": "Rule F17 is generating 41% false positives.",
        },
        "response": {"operation": "add"},
    },
    {
        "observation": {
            "domain": "fraud_detection",
            "task_name": "medium",
            "memory": [
                {"index": 0, "text": "chargebacks rose 18% WoW in LATAM"},
                {"index": 1, "text": "rule F17 has 41% false positives"},
            ],
            "new_message": "I watched a football match last night.",
        },
        "response": {"operation": "noop"},
    },
    {
        "observation": {
            "domain": "fraud_detection",
            "task_name": "medium",
            "memory": [
                {"index": 0, "text": "chargebacks rose 18% WoW in LATAM"},
                {"index": 1, "text": "rule F17 has 41% false positives"},
            ],
            "new_message": "Update: false positive rate is 14%, not 41%.",
        },
        "response": {"operation": "remove", "remove_index": 1},
    },
    {
        "observation": {
            "domain": "fraud_detection",
            "task_name": "medium",
            "memory": [{"index": 0, "text": "chargebacks rose 18% WoW in LATAM"}],
            "new_message": "Rule F17 false positive rate is 14%.",
        },
        "response": {"operation": "add"},
    },
    {
        "observation": {
            "domain": "ml_platform",
            "task_name": "hard",
            "memory": [
                {"index": 0, "text": "feature store write path times out at 3k rps"},
                {"index": 1, "text": "offline-online skew on user_age_bucket"},
                {"index": 2, "text": "nightly retrain job misses SLA by 45 min"},
                {"index": 3, "text": "need rollback plan for model v38"},
            ],
            "new_message": "My coffee machine broke this morning.",
        },
        "response": {"operation": "noop"},
    },
    {
        "observation": {
            "domain": "ml_platform",
            "task_name": "hard",
            "memory": [
                {"index": 0, "text": "feature store write path times out at 3k rps"},
                {"index": 1, "text": "offline-online skew on user_age_bucket"},
                {"index": 2, "text": "nightly retrain job misses SLA by 45 min"},
                {"index": 3, "text": "need rollback plan for model v38"},
            ],
            "new_message": "Inference service OOM appears only on model v38.",
        },
        "response": {"operation": "add"},
    },
    {
        "observation": {
            "domain": "ml_platform",
            "task_name": "hard",
            "memory": [
                {"index": 0, "text": "feature store write path times out at 3k rps"},
                {"index": 1, "text": "offline-online skew on user_age_bucket"},
                {"index": 2, "text": "nightly retrain job misses SLA by 45 min"},
                {"index": 3, "text": "need rollback plan for model v38"},
                {"index": 4, "text": "inference OOM only on model v38"},
            ],
            "new_message": "Correction: write path timeout starts at 1.8k rps.",
        },
        "response": {"operation": "remove", "remove_index": 0},
    },
    {
        "observation": {
            "domain": "ml_platform",
            "task_name": "hard",
            "memory": [
                {"index": 0, "text": "offline-online skew on user_age_bucket"},
                {"index": 1, "text": "nightly retrain job misses SLA by 45 min"},
                {"index": 2, "text": "need rollback plan for model v38"},
                {"index": 3, "text": "inference OOM only on model v38"},
            ],
            "new_message": "Feature store write path times out at 1.8k rps.",
        },
        "response": {"operation": "add"},
    },
    {
        "observation": {
            "domain": "analytics_pipeline",
            "task_name": "easy",
            "memory": [{"index": 0, "text": "ETL job failed at transform step"}],
            "new_message": "Error stack shows null pointer in normalize_country().",
        },
        "response": {"operation": "add"},
    },
    {
        "observation": {
            "domain": "analytics_pipeline",
            "task_name": "easy",
            "memory": [
                {"index": 0, "text": "ETL job failed at transform step"},
                {"index": 1, "text": "null pointer in normalize_country()"},
            ],
            "new_message": "Bought a new mechanical keyboard.",
        },
        "response": {"operation": "noop"},
    },
    {
        "observation": {
            "domain": "analytics_pipeline",
            "task_name": "easy",
            "memory": [
                {"index": 0, "text": "ETL job failed at transform step"},
                {"index": 1, "text": "null pointer in normalize_country()"},
            ],
            "new_message": "Correction: crash is in map_country_code(), not normalize_country().",
        },
        "response": {"operation": "remove", "remove_index": 1},
    },
    {
        "observation": {
            "domain": "analytics_pipeline",
            "task_name": "easy",
            "memory": [{"index": 0, "text": "ETL job failed at transform step"}],
            "new_message": "Crash is in map_country_code().",
        },
        "response": {"operation": "add"},
    },
    {
        "observation": {
            "domain": "release_engineering",
            "task_name": "medium",
            "memory": [
                {"index": 0, "text": "release candidate rc-12 failed smoke tests"},
                {"index": 1, "text": "payment service healthcheck timeout"},
                {"index": 2, "text": "rollback window ends at 22:00 UTC"},
            ],
            "new_message": "Family dinner is planned this Sunday.",
        },
        "response": {"operation": "noop"},
    },
    {
        "observation": {
            "domain": "release_engineering",
            "task_name": "medium",
            "memory": [
                {"index": 0, "text": "release candidate rc-12 failed smoke tests"},
                {"index": 1, "text": "payment service healthcheck timeout"},
                {"index": 2, "text": "rollback window ends at 22:00 UTC"},
            ],
            "new_message": "Rollback window was extended to 23:30 UTC.",
        },
        "response": {"operation": "remove", "remove_index": 2},
    },
    {
        "observation": {
            "domain": "release_engineering",
            "task_name": "medium",
            "memory": [
                {"index": 0, "text": "release candidate rc-12 failed smoke tests"},
                {"index": 1, "text": "payment service healthcheck timeout"},
            ],
            "new_message": "Rollback window ends at 23:30 UTC.",
        },
        "response": {"operation": "add"},
    },
    {
        "observation": {
            "domain": "release_engineering",
            "task_name": "medium",
            "memory": [
                {"index": 0, "text": "release candidate rc-12 failed smoke tests"},
                {"index": 1, "text": "payment service healthcheck timeout"},
                {"index": 2, "text": "rollback window ends at 23:30 UTC"},
            ],
            "new_message": "Root cause likely bad migration on orders table.",
        },
        "response": {"operation": "add"},
    },
    {
        "observation": {
            "domain": "oncall_notes",
            "task_name": "hard",
            "memory": [
                {"index": 0, "text": "cache cluster us-east-1 had 2 node flaps"},
                {"index": 1, "text": "api gateway rate limit raised to 4k rps"},
                {"index": 2, "text": "db replica lag peaked at 19s"},
                {"index": 3, "text": "customer ACME seeing timeout on /invoice"},
                {"index": 4, "text": "hotfix branch release/2026-04-25 prepared"},
                {"index": 5, "text": "SRE requested canary at 10 percent"},
            ],
            "new_message": "ACME timeout fixed after index on invoice_id.",
        },
        "response": {"operation": "add"},
    },
    {
        "observation": {
            "domain": "oncall_notes",
            "task_name": "hard",
            "memory": [
                {"index": 0, "text": "cache cluster us-east-1 had 2 node flaps"},
                {"index": 1, "text": "api gateway rate limit raised to 4k rps"},
                {"index": 2, "text": "db replica lag peaked at 19s"},
                {"index": 3, "text": "customer ACME seeing timeout on /invoice"},
                {"index": 4, "text": "hotfix branch release/2026-04-25 prepared"},
                {"index": 5, "text": "SRE requested canary at 10 percent"},
                {"index": 6, "text": "ACME timeout fixed after invoice_id index"},
            ],
            "new_message": "Correction: replica lag peak was 7s, not 19s.",
        },
        "response": {"operation": "remove", "remove_index": 2},
    },
    {
        "observation": {
            "domain": "oncall_notes",
            "task_name": "hard",
            "memory": [
                {"index": 0, "text": "cache cluster us-east-1 had 2 node flaps"},
                {"index": 1, "text": "api gateway rate limit raised to 4k rps"},
                {"index": 2, "text": "customer ACME seeing timeout on /invoice"},
                {"index": 3, "text": "hotfix branch release/2026-04-25 prepared"},
                {"index": 4, "text": "SRE requested canary at 10 percent"},
                {"index": 5, "text": "ACME timeout fixed after invoice_id index"},
            ],
            "new_message": "Replica lag peak was 7s.",
        },
        "response": {"operation": "add"},
    },
    {
        "observation": {
            "domain": "oncall_notes",
            "task_name": "hard",
            "memory": [
                {"index": 0, "text": "cache cluster us-east-1 had 2 node flaps"},
                {"index": 1, "text": "api gateway rate limit raised to 4k rps"},
                {"index": 2, "text": "customer ACME seeing timeout on /invoice"},
                {"index": 3, "text": "hotfix branch release/2026-04-25 prepared"},
                {"index": 4, "text": "SRE requested canary at 10 percent"},
                {"index": 5, "text": "ACME timeout fixed after invoice_id index"},
                {"index": 6, "text": "replica lag peak was 7s"},
            ],
            "new_message": "I listened to old jazz records at midnight.",
        },
        "response": {"operation": "noop"},
    },
]


def format_observation(observation: Dict[str, Any]) -> str:
    """Builds a stable prompt body with indexed memory entries."""
    memory_items = observation.get("memory", [])
    if not memory_items:
        memory_view = "(empty)"
    else:
        lines: List[str] = []
        for item in memory_items:
            idx = int(item["index"])
            txt = str(item["text"])
            lines.append(f"[{idx}] {txt}")
        memory_view = "\n".join(lines)

    domain = str(observation.get("domain", "unknown"))
    task_name = str(observation.get("task_name", "easy"))
    new_message = str(observation.get("new_message", ""))
    return (
        f"Domain: {domain}\n"
        f"Task difficulty: {task_name}\n"
        f"Memory entries ({len(memory_items)}):\n{memory_view}\n\n"
        f"New message:\n{new_message}\n\n"
        "Decide one action now."
    )


def action_to_json(action: Dict[str, Any]) -> str:
    """Canonical action JSON for supervised target text."""
    operation = str(action.get("operation", "noop"))
    if operation == "remove":
        return json.dumps(
            {
                "operation": "remove",
                "remove_index": int(action["remove_index"]),
            },
            ensure_ascii=False,
        )
    if operation == "add":
        return json.dumps({"operation": "add"}, ensure_ascii=False)
    return json.dumps({"operation": "noop"}, ensure_ascii=False)
