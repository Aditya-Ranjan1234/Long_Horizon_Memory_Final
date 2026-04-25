"""
Generate a large-scale Long Horizon Memory episodes dataset.

Output: server/episodes_large.json

Each episode keeps the same schema as episodes.json:
    {
        "episode_id": int,
        "conversation_domain": str,
        "difficulty": "easy" | "medium" | "hard",
        "string_relevant_messages": [
            {"text": str, "isRelevant": bool},
            ...
        ]
    }

Strategy:
- Hand-curated topic blueprints per domain produce coherent relevant strands
  via composable sentence patterns (subject + issue + observation + ...).
- A wide off-topic noise corpus is filtered by TF-IDF cosine similarity vs.
  the relevant content of each episode, so distractors are genuinely unrelated.
- Difficulty controls relevant fraction and noise volume.
- Deterministic given a seed; safe to re-run.

Usage:
    python build_large_episodes.py
    NUM_EPISODES_PER_DOMAIN=15 TARGET_LEN_MAX=160 python build_large_episodes.py
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


SEED = int(os.getenv("EPISODES_SEED", "1337"))
NUM_EPISODES_PER_DOMAIN = int(os.getenv("NUM_EPISODES_PER_DOMAIN", "10"))
TARGET_LEN_MIN = int(os.getenv("TARGET_LEN_MIN", "80"))
TARGET_LEN_MAX = int(os.getenv("TARGET_LEN_MAX", "120"))
# Optional cap on relevant items per episode. Useful to keep relevant counts
# below the env's MEMORY_CAPACITY so the recall signal stays clean.
MAX_RELEVANT_PER_EPISODE = int(os.getenv("MAX_RELEVANT_PER_EPISODE", "0"))
NOISE_COSINE_THRESHOLD = float(os.getenv("NOISE_COSINE_THRESHOLD", "0.05"))
OUTPUT_PATH = Path(__file__).with_name(os.getenv("OUTPUT_FILE", "episodes_large.json"))


DIFFICULTY_BY_DOMAIN: Dict[str, str] = {
    "education_ai": "easy",
    "career_guidance": "easy",
    "research_workflow": "easy",
    "customer_support_ops": "easy",
    "product_analytics": "easy",
    "software_debugging": "medium",
    "mobile_app_development": "medium",
    "data_engineering": "medium",
    "api_design": "medium",
    "music_production_workflow": "medium",
    "ai_system_design": "hard",
    "machine_learning_ops": "hard",
    "distributed_systems": "hard",
    "devops_infrastructure": "hard",
    "security_incident_response": "hard",
    "compiler_development": "hard",
    "game_engine_optimization": "hard",
    "nlp_system_design": "hard",
    "blockchain_development": "hard",
    "financial_fraud_detection": "hard",
    "photography_editing_pipeline": "hard",
    "sleep_quality_tracking": "hard",
    "logistics_optimization": "hard",
    "cybersecurity_audit": "hard",
    "entrepreneurship": "hard",
    "health_lifestyle": "medium",
}


DIFFICULTY_PROFILE: Dict[str, Dict[str, float]] = {
    "easy":   {"relevant_frac_min": 0.70, "relevant_frac_max": 0.85},
    "medium": {"relevant_frac_min": 0.50, "relevant_frac_max": 0.70},
    "hard":   {"relevant_frac_min": 0.30, "relevant_frac_max": 0.50},
}


# ── DOMAIN BLUEPRINTS ────────────────────────────────────────────────────────
# Each blueprint exposes composable slots. Sentences are assembled from these
# fragments by the message generator below. Add more fragments to scale the
# dataset further; the generator deduplicates and respects budgets.

DOMAIN_BLUEPRINTS: Dict[str, Dict[str, List[str]]] = {
    "software_debugging": {
        "subjects": [
            "the Python ETL job", "the auth microservice", "the report generator",
            "the payment gateway client", "the queue consumer", "the OCR pipeline",
        ],
        "issues": [
            "crashes intermittently after a few hours",
            "throws unicode decode errors on certain rows",
            "slows down sharply under bursty load",
            "leaks file handles between requests",
            "returns 500 on payloads above 10MB",
            "fails to recover after a database restart",
        ],
        "observations": [
            "logs show repeated retries with no backoff",
            "memory grows linearly with request count",
            "cpu spikes correlate with cache misses",
            "stack trace points to a recursive call in normalizer",
            "the failure only reproduces on Windows runners",
            "the heap snapshot reveals dangling closures",
        ],
        "attempts": [
            "we added structured logging around the failing call",
            "we tried reducing the worker pool size to four",
            "we patched the dependency to its 2.7.1 release",
            "we ran the regression suite against staging",
            "we instrumented the hot path with py-spy",
            "we replaced the json parser with orjson",
        ],
        "constraints": [
            "we cannot block the main loop for more than 50ms",
            "the service must run on a single 2GB container",
            "patches need to land before the Friday release",
            "rollback must be possible without DB migration",
            "external API quota is 600 requests per minute",
            "logs cannot include any customer PII",
        ],
        "decisions": [
            "we will introduce a circuit breaker around the external call",
            "we will route bulk requests through an async queue",
            "we will cap the in-memory buffer at one thousand items",
            "we will pin the pandas version to 2.1.4",
            "we will move the slow loop into a background worker",
            "we will add a feature flag to toggle the new path",
        ],
        "questions": [
            "is asyncio actually helping here or hiding the issue",
            "should we shard the queue per tenant",
            "do we need a persistent dead letter queue",
            "what is the safest way to add idempotency keys",
            "can we afford a full reindex during the maintenance window",
            "would moving to a managed queue reduce ops burden",
        ],
    },
    "ai_system_design": {
        "subjects": [
            "the multi-agent planner", "the tool-calling orchestrator",
            "the retrieval augmented copilot", "the speech-to-action agent",
            "the validator agent layer", "the hierarchical task graph",
        ],
        "issues": [
            "agents emit conflicting outputs on the same task",
            "tool calls loop without making progress",
            "context windows overflow on long sessions",
            "the planner produces unsupported steps",
            "evaluation rewards are noisy across rollouts",
            "shared memory drifts out of sync between agents",
        ],
        "observations": [
            "trace shows duplicate sub-task generation",
            "confidence scores collapse to near uniform",
            "the executor agent ignores deprecated tool versions",
            "vector retrieval returns near-duplicate snippets",
            "feedback loops oscillate without converging",
            "response latency grows with reasoning depth",
        ],
        "attempts": [
            "we added a critique step before execution",
            "we tried temperature 0.3 with top-p 0.9",
            "we cached intermediate plans by hash",
            "we deduplicated retrieval chunks by semantic id",
            "we constrained the action schema with a grammar",
            "we capped reasoning depth to four hops",
        ],
        "constraints": [
            "the system must respect a 4-second wall-clock budget",
            "we cannot fine-tune the base model",
            "agent calls must be auditable per session",
            "the memory store must be serializable to disk",
            "we must keep cost per session under five cents",
            "all tool calls must pass an allowlist",
        ],
        "decisions": [
            "we will add a confidence-weighted aggregator",
            "we will introduce a planning cache keyed by goal",
            "we will route validation to a smaller cheaper model",
            "we will store partial plans for resume after crash",
            "we will impose a max retries policy per tool",
            "we will move retrieval to a hybrid sparse-plus-dense index",
        ],
        "questions": [
            "do we benefit from a dedicated critic agent",
            "should planning be tree-based or sequential",
            "is shared scratchpad better than message passing",
            "how do we attribute reward to a single agent",
            "should validation block or run in parallel",
            "can we offload long-term memory to a vector db",
        ],
    },
    "machine_learning_ops": {
        "subjects": [
            "the nightly training pipeline", "the feature store",
            "the model registry", "the inference autoscaler",
            "the embedding refresh job", "the drift monitor",
        ],
        "issues": [
            "training fails after the data validation step",
            "feature backfill takes longer than the SLA",
            "online model latency creeps up over a day",
            "drift alerts fire on benign weekend traffic",
            "shadow models silently diverge from production",
            "the autoscaler oscillates between min and max replicas",
        ],
        "observations": [
            "validation loss diverges in the third epoch",
            "feature freshness lag peaks at 18 minutes",
            "p99 latency aligns with cold start events",
            "feature skew between offline and online is 4 percent",
            "rollout 38 has 2 percent more 5xx",
            "logits drift slowly on the long tail segment",
        ],
        "attempts": [
            "we added a row count guard to the validator",
            "we precomputed daily aggregates ahead of time",
            "we warmed up replicas with shadow traffic",
            "we recalibrated the drift detector thresholds",
            "we constrained training data to the last 30 days",
            "we sampled by user cohort to reduce variance",
        ],
        "constraints": [
            "training jobs must complete inside a 6 hour window",
            "the feature store budget caps writes at 5k qps",
            "deployments need 24 hour shadow before promotion",
            "we must keep model size under 250 megabytes",
            "rollbacks must be one-click from any dashboard",
            "audit logs must include input feature hashes",
        ],
        "decisions": [
            "we will introduce a canary cohort before full ramp",
            "we will partition the feature store by namespace",
            "we will retire the v37 ensemble on Friday",
            "we will add a circuit breaker around the slow feature",
            "we will collect per-segment evaluation metrics",
            "we will move embedding refresh to a streaming job",
        ],
        "questions": [
            "is online learning safe for the fraud model",
            "should we monitor embeddings or only outputs",
            "do we need a separate eval cluster",
            "is there a cleaner way to attribute drift to inputs",
            "should we cache predictions for repeat visitors",
            "how do we benchmark cold start under real load",
        ],
    },
    "security_incident_response": {
        "subjects": [
            "the production cluster", "the API gateway",
            "the secrets manager", "the user account service",
            "the audit logging pipeline", "the SSO bridge",
        ],
        "issues": [
            "we observed unusual login patterns from a new ASN",
            "an internal service is making egress to an unknown host",
            "the WAF triggered on suspicious payloads at 02:00 UTC",
            "a compromised token may be in active use",
            "the audit trail has gaps during the deploy window",
            "an alert keeps firing on the same indicator",
        ],
        "observations": [
            "egress traffic doubled on port 4444 for 11 minutes",
            "five accounts had successful logins from new countries",
            "an environment variable contained a stale credential",
            "the SIEM correlation flagged three related events",
            "a privileged action was taken outside change windows",
            "the indicator overlaps with a known IoC list",
        ],
        "attempts": [
            "we rotated the affected service account credentials",
            "we revoked all tokens issued before 14:00 UTC",
            "we restricted egress to a known good allowlist",
            "we increased logging verbosity on the auth service",
            "we enabled stepped-up MFA for at-risk cohorts",
            "we ran the indicator against the threat intel feed",
        ],
        "constraints": [
            "containment must not break customer-facing flows",
            "we cannot reset passwords for VIP accounts without sign-off",
            "all changes must be tracked in the incident log",
            "any rotation must respect dependent service downstream",
            "evidence preservation must precede any wipe",
            "comms cannot leak unconfirmed attribution",
        ],
        "decisions": [
            "we will enroll the affected scope into stepped-up auth",
            "we will roll forward to the patched gateway image",
            "we will quarantine the suspicious node for forensics",
            "we will require break-glass approval for restoration",
            "we will publish a postmortem within five business days",
            "we will add a rule for this indicator to the WAF",
        ],
        "questions": [
            "can we attribute the activity to a single principal",
            "is the scope limited to production or also stage",
            "do we have integrity confirmation for the audit log",
            "should we notify customers proactively",
            "what is the blast radius of the leaked token",
            "are we missing detection on adjacent services",
        ],
    },
    "data_engineering": {
        "subjects": [
            "the daily ingestion DAG", "the change data capture stream",
            "the analytics warehouse", "the schema registry",
            "the dimensional model", "the data quality framework",
        ],
        "issues": [
            "ingestion lag spikes after midnight UTC",
            "schema evolution silently broke a downstream job",
            "duplicate rows appear after retries",
            "partition pruning is no longer effective",
            "warehouse cost has grown 30 percent month over month",
            "late-arriving events corrupt rolling aggregates",
        ],
        "observations": [
            "the slow query plan does a full table scan",
            "watermark progression stalls on a small partition",
            "kafka consumer lag grows to 1.2 million",
            "small files multiply faster than compaction can run",
            "row counts diverge between staging and production",
            "metadata services see triple query rate on Mondays",
        ],
        "attempts": [
            "we changed clustering keys on the largest table",
            "we added watermark-aware deduplication",
            "we increased compaction frequency to hourly",
            "we tightened schema contracts on the producer side",
            "we precomputed expensive joins into a daily table",
            "we partitioned by event_date and tenant_id",
        ],
        "constraints": [
            "we cannot break the existing report contract",
            "warehouse credits must stay under monthly budget",
            "transformations must be incremental where possible",
            "all tables must have row-level lineage attached",
            "PII columns must remain masked in dev",
            "we must support backfill for at least 90 days",
        ],
        "decisions": [
            "we will move the heavy job to a reserved warehouse",
            "we will introduce data contracts for the top three sources",
            "we will deprecate the legacy mart by quarter end",
            "we will add row-level expectations in the QA layer",
            "we will swap the merge strategy on the slow target",
            "we will document partition keys in the catalog",
        ],
        "questions": [
            "is delta the right format for this access pattern",
            "should we trigger backfills automatically on schema change",
            "do we need a separate freshness SLA per source",
            "how do we attribute warehouse cost to teams",
            "can we move quality checks closer to ingestion",
            "should we enforce contracts in CI",
        ],
    },
    "compiler_development": {
        "subjects": [
            "the parser front end", "the type checker", "the IR builder",
            "the optimization pass manager", "the codegen backend",
            "the standard library bridge",
        ],
        "issues": [
            "the parser fails on nested generics with default values",
            "type inference loops on recursive structures",
            "the SSA pass corrupts loop induction variables",
            "codegen emits the wrong jump target for switch",
            "binary size grew 12 percent after the last merge",
            "incremental rebuilds rerun more than expected",
        ],
        "observations": [
            "the regression set narrows to three small files",
            "phi nodes get duplicated after dead code elimination",
            "instruction selection picks suboptimal patterns",
            "compile time peaks during constant folding",
            "the memory leak appears only with optimization level two",
            "diagnostics quality regresses on macro expansions",
        ],
        "attempts": [
            "we bisected the regression to a single commit",
            "we added an SSA verifier between passes",
            "we wrote IR diff tests for the failing fixtures",
            "we restricted inlining heuristics under deep templates",
            "we tuned the cost model for branch heavy code",
            "we cached parsed AST fragments across modules",
        ],
        "constraints": [
            "the front end must remain stable for downstream tools",
            "we must keep IR compatibility with the previous version",
            "test runtime cannot exceed the existing budget",
            "we cannot regress diagnostics quality",
            "binary backwards compatibility must hold",
            "warnings as errors must remain enabled in CI",
        ],
        "decisions": [
            "we will gate the optimization behind a pragma",
            "we will revert the inliner change while we triage",
            "we will add fuzzing for the type checker",
            "we will land the new pass behind a feature flag",
            "we will refactor the cost model in a separate PR",
            "we will track binary size in CI artifacts",
        ],
        "questions": [
            "is region based inference worth the complexity",
            "should we move dataflow analysis into MLIR",
            "do we need a dedicated ABI tester",
            "can we adopt incremental parsing without an ABI break",
            "what is the right interface for plugin passes",
            "should we expose IR snapshots in the debugger",
        ],
    },
    "distributed_systems": {
        "subjects": [
            "the consensus layer", "the gossip membership", "the storage tier",
            "the request router", "the rebalancer", "the lease manager",
        ],
        "issues": [
            "tail latency spikes during leader election",
            "replication lag grows after a network partition",
            "rebalance moves shards in a thundering herd",
            "cross-zone traffic doubles unexpectedly",
            "stale reads appear during follower failover",
            "lease renewal misses on slow garbage collection",
        ],
        "observations": [
            "GC pauses correlate with election storms",
            "shard count peaks at 1.4 million during rebalance",
            "cross-zone traffic peaks at 3.7 GB per minute",
            "follower handoff window averages 2.4 seconds",
            "p99.9 read latency triples for 11 minutes",
            "metric cardinality explodes on shard splits",
        ],
        "attempts": [
            "we tightened election timeout windows",
            "we throttled rebalance moves to 8 per minute",
            "we pinned hot shards to a dedicated node group",
            "we adopted hedged reads on the slow tier",
            "we increased lease renewal frequency",
            "we added jitter to retry backoff",
        ],
        "constraints": [
            "we cannot exceed 10 percent additional cross-zone bandwidth",
            "consensus quorum must remain available across one zone outage",
            "client SDKs cannot be upgraded for at least one quarter",
            "we cannot change the on-disk format this release",
            "rolling restarts must be safe for read traffic",
            "max blast radius per change is one cell",
        ],
        "decisions": [
            "we will bound rebalance velocity by a global limiter",
            "we will publish a lease handoff API for clients",
            "we will partition the metric cardinality by tenant",
            "we will deploy the new election quickfix in a canary cell",
            "we will adopt hedged requests on read paths",
            "we will hide the new behavior behind a flag for two weeks",
        ],
        "questions": [
            "should we move to a leaderless replication model",
            "is hinted handoff safe under our consistency contract",
            "do we need a chaos engineering rotation",
            "how do we measure tail latency under partial outages",
            "can we shrink quorum size for one read path",
            "should clients be aware of zone topology",
        ],
    },
    "mobile_app_development": {
        "subjects": [
            "the iOS onboarding flow", "the Android background sync",
            "the offline cache", "the push notification pipeline",
            "the deep link router", "the analytics SDK wrapper",
        ],
        "issues": [
            "cold start time regressed by 600 milliseconds",
            "background sync drains battery on certain devices",
            "deep links sometimes route to the previous screen",
            "push delivery rate dipped to 88 percent on Tuesday",
            "the offline cache evicts prematurely on low-end devices",
            "the SDK adds an extra 5 megabytes to the binary",
        ],
        "observations": [
            "trace shows duplicate API calls on resume",
            "battery delta is largest on devices with weak signal",
            "navigation stack contains stale view controllers",
            "Firebase logs reveal token refresh storms",
            "cache hit rate drops below 60 percent after upgrade",
            "binary growth is mostly in resource bundles",
        ],
        "attempts": [
            "we deferred analytics initialization until after first frame",
            "we batched background tasks behind a single coalescer",
            "we rewrote the deep link parser with explicit states",
            "we throttled push retries with exponential backoff",
            "we added LRU sizing to the offline cache",
            "we trimmed unused asset variants",
        ],
        "constraints": [
            "we must preserve compatibility with iOS 14",
            "Android startup target is under 1.2 seconds on midrange",
            "weekly battery regression must not exceed 1 percent",
            "binary growth budget per release is 800 kilobytes",
            "deep link routes must be backward compatible",
            "third party SDK count cannot grow this quarter",
        ],
        "decisions": [
            "we will gate the new sync engine behind remote config",
            "we will move analytics to a deferred queue",
            "we will adopt URLSession for legacy networking",
            "we will cache thumbnails at multiple resolutions",
            "we will require code-level review for any SDK addition",
            "we will track binary size via CI gating",
        ],
        "questions": [
            "is Compose Multiplatform worth a small pilot",
            "do we need a per-region deep link table",
            "should we adopt a single networking layer cross-platform",
            "how do we benchmark battery in CI reliably",
            "is module level lazy loading safe for our flows",
            "should we keep separate iOS and Android analytics SDKs",
        ],
    },
    "devops_infrastructure": {
        "subjects": [
            "the production Kubernetes cluster", "the CI pipeline",
            "the secrets vault", "the terraform monorepo",
            "the observability stack", "the cost dashboard",
        ],
        "issues": [
            "build times grew from 9 to 21 minutes",
            "the node autoscaler thrashes during peak hours",
            "secrets rotation broke two stateful jobs",
            "metrics ingestion misses scrape windows on Mondays",
            "dashboard p95 query time exceeds three seconds",
            "the monthly bill is up 28 percent quarter over quarter",
        ],
        "observations": [
            "image pull time dominates job startup",
            "the autoscaler reacts to flapping node pressure",
            "secret renewal hits rate limit at 09:00 UTC",
            "scrape duration aligns with cron heavy minutes",
            "metric cardinality grew by 46 percent",
            "spend skew is concentrated in two namespaces",
        ],
        "attempts": [
            "we cached docker layers across pipeline stages",
            "we tuned cluster autoscaler thresholds",
            "we paginated secrets renewal across windows",
            "we throttled ingestion at the receiver",
            "we promoted hot dashboards to materialized views",
            "we added cost attribution labels at scheduler time",
        ],
        "constraints": [
            "production must remain on the LTS Kubernetes release",
            "downtime windows must be coordinated with SRE on call",
            "deploys must be canaryable per region",
            "secrets cannot live unencrypted at rest",
            "log retention must remain at 30 days minimum",
            "any IAM change requires two-person approval",
        ],
        "decisions": [
            "we will move the build cache to a regional bucket",
            "we will roll out spot nodes to non-critical pools",
            "we will add per-namespace cost budgets",
            "we will split the monolithic terraform repo by domain",
            "we will add a synthetic probe per critical path",
            "we will enforce cardinality limits via metrics policy",
        ],
        "questions": [
            "should we move CI to ARM runners",
            "is GitOps the right path for our infra updates",
            "do we need fine-grained IAM per service account",
            "can we afford a separate observability cluster",
            "should we adopt service mesh for north-south traffic",
            "is finops a part-time or full-time function for us",
        ],
    },
    "blockchain_development": {
        "subjects": [
            "the smart contract suite", "the on-chain indexer",
            "the wallet onboarding flow", "the bridge contract",
            "the staking module", "the RPC infrastructure",
        ],
        "issues": [
            "gas usage spiked after the recent storage layout change",
            "the indexer falls behind during high block production",
            "wallet creation rate is below conversion target",
            "bridge processing has stalled on small denominations",
            "staking rewards distribution mismatched by 0.3 percent",
            "RPC nodes restart unpredictably under heavy load",
        ],
        "observations": [
            "storage slot collisions occur on upgraded contracts",
            "indexer lag peaks at 240 blocks on Mondays",
            "drop-off concentrates after the seed phrase prompt",
            "the bridge fee model penalizes small transfers heavily",
            "the reward script rounds wei in the wrong direction",
            "RPC restarts cluster around peak mempool growth",
        ],
        "attempts": [
            "we packed two booleans into one storage slot",
            "we sharded the indexer by contract namespace",
            "we shortened the seed phrase confirmation flow",
            "we adjusted bridge fee tiers for small transfers",
            "we re-derived the reward formula with bigint math",
            "we tuned the RPC connection pool size",
        ],
        "constraints": [
            "we cannot break ABI for downstream integrators",
            "audits must precede any mainnet deployment",
            "wallet UX cannot push users through more than four steps",
            "bridge throughput target is 90 percent of source chain",
            "reward errors must round in favor of users",
            "RPC SLA must remain at 99.95 percent availability",
        ],
        "decisions": [
            "we will deploy a proxy with explicit slot reservations",
            "we will run two indexer fleets and reconcile",
            "we will add a recovery UX for partial onboarding",
            "we will lower the minimum bridgeable amount",
            "we will add unit tests with bigint property checks",
            "we will provision dedicated nodes for partner traffic",
        ],
        "questions": [
            "is L2 the right deployment target for this product",
            "should we open-source the indexer for community use",
            "do we need a multi-chain abstraction layer",
            "how do we audit upgrades safely under proxies",
            "is account abstraction ready for our user base",
            "should we offer a hosted RPC tier",
        ],
    },
    "education_ai": {
        "subjects": [
            "the question generation pipeline", "the answer checker",
            "the difficulty calibrator", "the grading rubric system",
            "the personalization engine", "the lecture summarizer",
        ],
        "issues": [
            "questions become repetitive across chunks",
            "answers receive partial credit inconsistently",
            "difficulty drifts toward easier items over time",
            "grading misses application-style questions",
            "personalization recommends content out of order",
            "summaries lose key formula derivations",
        ],
        "observations": [
            "duplicate stems appear within the same chapter",
            "rubric overlap causes double counting",
            "the calibrator favors recall over reasoning items",
            "graders disagree on partial credit boundaries",
            "students with low scores see harder content",
            "the summarizer truncates when notes exceed four pages",
        ],
        "attempts": [
            "we added semantic dedup over question stems",
            "we tightened the grading rubric language",
            "we recalibrated difficulty using IRT estimates",
            "we added an application-only question generator",
            "we reranked items by mastery probability",
            "we chunked notes by topical headings",
        ],
        "constraints": [
            "the system must run on a small classroom budget",
            "questions must align with the official syllabus",
            "we cannot rely on a paid LLM for the offline path",
            "data must be retained per district policy",
            "summaries must include latex when appropriate",
            "the system must be operable by non-engineer staff",
        ],
        "decisions": [
            "we will publish a teacher dashboard for content review",
            "we will introduce a peer review queue for edge items",
            "we will offer per-student difficulty curves",
            "we will add bloom-level tags to every item",
            "we will store rubric annotations alongside answers",
            "we will translate prompts for the bilingual cohort",
        ],
        "questions": [
            "is RAG worth maintaining for short note chunks",
            "should we keep human-in-the-loop for hard items",
            "do we need item exposure controls per cohort",
            "how do we evaluate generated explanations fairly",
            "should we offer a spaced repetition mode",
            "is offline grading viable on classroom hardware",
        ],
    },
    "career_guidance": {
        "subjects": [
            "the ML versus full-stack decision", "the open source portfolio",
            "the interview preparation plan", "the mentorship search",
            "the side project pipeline", "the long-term career roadmap",
        ],
        "issues": [
            "I enjoy two paths but have shallow depth in each",
            "open source contributions are sparse this quarter",
            "interview rejections cluster around system design",
            "mentorship attempts have not led to repeat sessions",
            "side projects often stall after the prototype",
            "I cannot commit to a single specialization yet",
        ],
        "observations": [
            "my project graphs follow a pattern of early peaks",
            "interview feedback emphasizes scoping and trade-offs",
            "mentor responses fade after the first introduction",
            "GitHub activity is concentrated on weekends only",
            "I keep restarting topics rather than completing one",
            "research summaries pile up without follow-through",
        ],
        "attempts": [
            "I built two projects to compare ML and product work",
            "I scheduled mock system design sessions weekly",
            "I committed to one open source repo for two months",
            "I joined a structured book club on distributed systems",
            "I tracked study hours by topic to spot drift",
            "I committed to publishing a small write-up monthly",
        ],
        "constraints": [
            "I cannot relocate before next academic break",
            "I have at most ten focused hours per week",
            "I want to keep options open for at least a year",
            "I prefer roles with both research and shipping",
            "I want to avoid late night on-call rotations",
            "I need work that allows steady salary growth",
        ],
        "decisions": [
            "I will pick one specialization for the next six months",
            "I will draft a portfolio aimed at applied ML roles",
            "I will publish weekly notes on system design progress",
            "I will commit to one mentor with structured cadence",
            "I will reserve one weekend per month for resume work",
            "I will attend one in-person meetup per quarter",
        ],
        "questions": [
            "should I aim for product or platform teams first",
            "how do I prove depth without a published paper",
            "is graduate school worth it for applied roles",
            "should I optimize for breadth or focus this year",
            "what is the right cadence for portfolio updates",
            "how do I evaluate when to switch teams",
        ],
    },
    "research_workflow": {
        "subjects": [
            "the literature review queue", "the experiment tracking spreadsheet",
            "the latex paper draft", "the related work synthesis",
            "the reproducibility script bundle", "the citation manager",
        ],
        "issues": [
            "important papers slip through the weekly review",
            "experiment names diverge from the actual configs",
            "draft sections drift out of sync with figures",
            "related work coverage is uneven across themes",
            "scripts fail to reproduce results from last quarter",
            "the citation manager imports broken bibtex entries",
        ],
        "observations": [
            "the queue grows by twelve papers per week on average",
            "config files have stale defaults from prior runs",
            "section figures use inconsistent units",
            "two themes dominate citations while a third is sparse",
            "results differ when run on a new GPU model",
            "ten percent of bibtex entries have malformed URLs",
        ],
        "attempts": [
            "I rewrote the queue with priority tags by venue",
            "I introduced a single source of truth for configs",
            "I added a continuous build for the latex paper",
            "I made a related work coverage matrix per section",
            "I pinned the GPU drivers in the reproducibility bundle",
            "I scripted bibtex sanity checks before commit",
        ],
        "constraints": [
            "the deadline is the camera-ready in twelve days",
            "experiments must use the shared cluster fairly",
            "data licensing forbids redistribution of raw audio",
            "compute budget caps a single run at four hours",
            "the writeup must follow the venue style guide",
            "co-author reviews must not block more than two days",
        ],
        "decisions": [
            "we will freeze the experiment grid by Friday",
            "we will move figure code into a shared module",
            "we will publish anonymized configs alongside the paper",
            "we will add a hash for each artifact in the appendix",
            "we will rotate review responsibilities across sections",
            "we will tag each paragraph with its supporting evidence",
        ],
        "questions": [
            "is preregistration helpful for our experimental claims",
            "should we share negative results in the appendix",
            "do we need a separate pre-print server upload",
            "is paired t-test the right comparison here",
            "should we ablate by component or by hyperparameter",
            "how do we keep the related work fresh after submission",
        ],
    },
    "customer_support_ops": {
        "subjects": [
            "the ticket triage queue", "the macros library",
            "the escalation tree", "the SLA dashboard",
            "the after-hours rotation", "the agent training plan",
        ],
        "issues": [
            "first response time slipped above the SLA on Tuesday",
            "macros do not match the new product taxonomy",
            "escalations skip the regional tier",
            "the dashboard misclassifies a third of resolved tickets",
            "after-hours volume doubled without scheduled coverage",
            "newer agents need longer to handle complex issues",
        ],
        "observations": [
            "p90 response time is 18 minutes versus 12 SLA",
            "macro match rate dropped to 48 percent",
            "the skip pattern correlates with weekend shifts",
            "the dashboard miscount stems from a tag rename",
            "tier-1 has 62 percent of after-hours load",
            "training scores plateau around the third week",
        ],
        "attempts": [
            "we re-tagged inbound tickets to the new taxonomy",
            "we updated the top twelve macros for v3 features",
            "we restored the regional escalation step",
            "we patched dashboard mappings for the renamed tag",
            "we added an after-hours rotation across two regions",
            "we paired senior agents with new hires for shadowing",
        ],
        "constraints": [
            "we cannot dedicate more than 12 percent of headcount to training",
            "agents must follow the brand voice guide for replies",
            "after-hours coverage must respect labor regulations",
            "tooling changes must not break legacy macros",
            "any data export must respect customer privacy",
            "queue routing must stay configurable by region",
        ],
        "decisions": [
            "we will publish a v3 macros sprint with QA review",
            "we will rebuild the escalation tree by region",
            "we will roll out a dashboard alert for tag drift",
            "we will hire two senior agents for after-hours",
            "we will add a knowledge base nudge in the agent UI",
            "we will track coaching time per new agent",
        ],
        "questions": [
            "is conversational AI ready to handle tier-1 fully",
            "should we measure CSAT per macro family",
            "do we need separate queues for retention and growth",
            "how do we evaluate agent expertise objectively",
            "should we adopt a knowledge graph for replies",
            "is auto-translation safe for sensitive accounts",
        ],
    },
    "product_analytics": {
        "subjects": [
            "the activation funnel", "the retention curves",
            "the experimentation platform", "the dashboard suite",
            "the cohort tagging system", "the event taxonomy",
        ],
        "issues": [
            "activation rate dropped 4 percent week over week",
            "retention curves flatten earlier than expected",
            "experiment power is insufficient for niche cohorts",
            "dashboards diverge between teams using the same metric",
            "cohort tags are inconsistent across platforms",
            "new events take weeks to land in the warehouse",
        ],
        "observations": [
            "drop-off concentrates at the email confirmation step",
            "the 30 day retention curve has a steeper second week",
            "the niche cohort accounts for 6 percent of users",
            "two teams compute DAU using different filters",
            "the cohort tag mismatch is mostly Android specific",
            "the event registration cycle takes about 11 days",
        ],
        "attempts": [
            "we shortened the email confirmation message",
            "we re-segmented users by device language",
            "we ran a paired analysis on duplicate dashboards",
            "we centralized the DAU formula in the metrics catalog",
            "we backfilled tags for the past 60 days",
            "we added a streamlined event registration form",
        ],
        "constraints": [
            "experiments must respect privacy review guidelines",
            "metric definitions must be reviewed by data governance",
            "we cannot ship dashboards without owner accountability",
            "tagging changes must be tested against legacy queries",
            "stakeholder requests need a triaged backlog",
            "raw exports must remain access-controlled",
        ],
        "decisions": [
            "we will deprecate three duplicate dashboards by quarter end",
            "we will publish a single source DAU formula",
            "we will adopt a lightweight tagging schema in CI",
            "we will offer experiment power calculators in the platform",
            "we will add a dashboard ownership column",
            "we will track event lead times as a top-line metric",
        ],
        "questions": [
            "should we adopt incrementality as default for experiments",
            "do we need a metrics review board",
            "should every dashboard have an owner and SLA",
            "is there a single tool that fits all teams",
            "how do we baseline cohort comparisons fairly",
            "should we open up self-serve queries to non-engineers",
        ],
    },
    "api_design": {
        "subjects": [
            "the public REST endpoints", "the GraphQL schema",
            "the rate limit policy", "the auth model",
            "the SDK generation pipeline", "the deprecation strategy",
        ],
        "issues": [
            "REST endpoints diverge in pagination conventions",
            "GraphQL queries occasionally exceed depth limits",
            "rate limit headers are inconsistent across endpoints",
            "auth scopes feel coarse for partner integrations",
            "SDK releases lag behind backend changes",
            "deprecation timelines are not clearly communicated",
        ],
        "observations": [
            "five endpoints use cursor while three use offset",
            "depth-limited queries spike at 14:00 UTC",
            "rate limit reset uses different units across services",
            "partner accounts request fine-grained scopes weekly",
            "the SDK release cycle averages two weeks behind",
            "deprecation notices are buried in changelogs",
        ],
        "attempts": [
            "we standardized on cursor pagination across endpoints",
            "we capped query depth to a tested level",
            "we unified rate limit headers behind a small middleware",
            "we drafted a scope catalog for partners",
            "we automated SDK regeneration on schema commits",
            "we added an explicit deprecation policy",
        ],
        "constraints": [
            "breaking changes require a 6 month deprecation window",
            "any new endpoint needs an OpenAPI spec on day one",
            "the auth flow cannot require manual key rotation",
            "client SDK size must not increase by 10 percent",
            "rate limits must be communicated in headers",
            "every public endpoint must have integration tests",
        ],
        "decisions": [
            "we will publish a versioned changelog with timelines",
            "we will move pagination to cursor-only for all v3",
            "we will adopt a partner sandbox with synthetic data",
            "we will offer typed SDKs for the top three languages",
            "we will require RFC review for any new endpoint",
            "we will add a rate limit dashboard for partners",
        ],
        "questions": [
            "is GraphQL still right for our partner surface",
            "should we adopt OAuth2 device flow for SDKs",
            "do we need versioning at the URL or header level",
            "should we open up bulk endpoints to enterprise tiers",
            "is hypermedia worth the complexity for our team",
            "how do we measure API DX per partner cohort",
        ],
    },
    "music_production_workflow": {
        "subjects": [
            "the DAW project template", "the plugin chain configuration",
            "the mastering preset bank", "the collaborative file sync",
            "the MIDI controller mapping", "the loudness target",
        ],
        "issues": [
            "DAW projects open with missing plugin instances",
            "plugin chains spike CPU on certain track presets",
            "mastering presets have inconsistent loudness output",
            "collaborative sync corrupts session backups occasionally",
            "MIDI mappings reset after firmware updates",
            "stems exported with peaks above the streaming target",
        ],
        "observations": [
            "missing plugins always cluster around third party reverbs",
            "CPU spikes correlate with high oversampling settings",
            "preset loudness varies by 2 LUFS across versions",
            "session corruption appears after concurrent saves",
            "MIDI maps are wiped only after major firmware versions",
            "peak excess is highest on the bass-heavy stems",
        ],
        "attempts": [
            "we created a project template with verified plugins only",
            "we lowered oversampling to two times except on master bus",
            "we re-rendered presets at consistent target loudness",
            "we routed sync through a single source-of-truth folder",
            "we exported MIDI mappings as JSON for restore",
            "we added a true peak limiter to all bass stems",
        ],
        "constraints": [
            "session size must stay portable across collaborators",
            "renders must meet streaming loudness specifications",
            "plugins must be available on all collaborator machines",
            "we cannot rely on a single online sync provider",
            "we must preserve compatibility with older session backups",
            "we must keep CPU headroom for live recording",
        ],
        "decisions": [
            "we will publish a plugin manifest with version locks",
            "we will adopt LUFS-targeted mastering presets",
            "we will keep collaborative sessions in versioned snapshots",
            "we will document MIDI mappings per controller model",
            "we will limit oversampling to bus level only",
            "we will share session reference files monthly",
        ],
        "questions": [
            "should we move to a cloud session host",
            "do we need a centralized loudness reference track",
            "is hardware controller mapping worth the maintenance",
            "should we standardize on one plugin vendor for reverbs",
            "is rendering on a render farm worth the setup",
            "should we open the project template to interns",
        ],
    },
    "photography_editing_pipeline": {
        "subjects": [
            "the raw import workflow", "the catalog backups",
            "the color grading presets", "the export configurations",
            "the metadata tagging policy", "the print proof process",
        ],
        "issues": [
            "raw imports occasionally lose camera profile metadata",
            "catalog backups grow faster than disk capacity",
            "color presets clip highlights on overexposed shots",
            "the export queue stalls during night batches",
            "metadata tags drift across photographers",
            "print proofs differ from screen calibration",
        ],
        "observations": [
            "missing profiles cluster around new lens releases",
            "backup growth aligns with raw plus xmp duplication",
            "highlight clipping appears mostly above ISO 1600",
            "export stalls correlate with the auto-tagging pass",
            "tag drift involves keyword case and synonyms",
            "print prints run warmer than monitor output",
        ],
        "attempts": [
            "we manually re-applied profiles to affected sessions",
            "we deduplicated xmp during incremental backups",
            "we adjusted highlight rolloff in the preset family",
            "we deferred auto-tagging to a daytime job",
            "we standardized keywords through a controlled vocabulary",
            "we recalibrated monitors to match the print profile",
        ],
        "constraints": [
            "we cannot lose original captures under any condition",
            "color grading must be reversible",
            "exports must align with publication specifications",
            "tag policies need to be friendly to the entire team",
            "monitor calibration must be auditable",
            "print runs must keep cost predictable",
        ],
        "decisions": [
            "we will adopt a strict folder structure per shoot",
            "we will upload backups to two distinct providers",
            "we will publish a preset family per genre",
            "we will move auto-tagging to a queue with priorities",
            "we will document a controlled vocabulary",
            "we will calibrate monitors monthly",
        ],
        "questions": [
            "is RAW plus DNG the right archival approach",
            "should we adopt a cloud catalog for collaboration",
            "do we need a redundant offsite backup",
            "is auto-tagging accuracy worth the cost",
            "should we offer print-on-demand integrations",
            "is a profile per photographer better than a shared preset",
        ],
    },
    "sleep_quality_tracking": {
        "subjects": [
            "the sleep tracking watch", "the bedtime routine",
            "the caffeine ledger", "the screen time policy",
            "the bedroom environment", "the recovery score",
        ],
        "issues": [
            "deep sleep duration regressed last week",
            "I struggle to fall asleep before midnight",
            "afternoon caffeine seems to push my bedtime later",
            "screen time before bed correlates with restlessness",
            "bedroom temperature varies more than expected",
            "the recovery score swings without clear cause",
        ],
        "observations": [
            "deep sleep stages dropped by 18 minutes on average",
            "sleep onset latency averages 36 minutes",
            "caffeine after 16:00 correlates with later sleep",
            "device usage past 23:00 correlates with awakenings",
            "temperature spikes by 2 Celsius in early morning",
            "recovery dips overlap with poor previous-day diet",
        ],
        "attempts": [
            "I cut caffeine after 14:00 for two weeks",
            "I introduced a wind-down routine starting at 21:30",
            "I switched to a low-temperature mattress topper",
            "I activated grayscale on devices after 22:00",
            "I logged daily activity to find correlations",
            "I limited dinner carbs on poor recovery days",
        ],
        "constraints": [
            "I cannot adjust schedule on early meeting days",
            "I want changes that are sustainable over months",
            "I do not want to take additional supplements",
            "I prefer to keep one social outing per weekend",
            "I cannot reduce work hours during release weeks",
            "I want fewer manual log entries over time",
        ],
        "decisions": [
            "I will keep caffeine cutoff at 14:00 for one quarter",
            "I will treat lighting as part of the wind-down routine",
            "I will track temperature trends weekly",
            "I will trial a 30-minute walk after dinner",
            "I will limit late screens on weekdays",
            "I will book recovery weekends after release sprints",
        ],
        "questions": [
            "is HRV variation a useful daily decision input",
            "should I track diet alongside sleep",
            "how do I separate stress effects from caffeine",
            "is short napping helpful or counterproductive",
            "should I move workouts away from late evening",
            "what is a meaningful recovery threshold",
        ],
    },
    "logistics_optimization": {
        "subjects": [
            "the last-mile routing engine", "the dispatch dashboard",
            "the warehouse pick path", "the cross-dock policy",
            "the carrier mix", "the return processing pipeline",
        ],
        "issues": [
            "delivery exceptions concentrate in two zip codes",
            "dispatch decisions look stale during traffic spikes",
            "the warehouse pick path is longer than necessary",
            "cross-dock misses peak afternoon flow",
            "the carrier mix favors slower options for high-priority parcels",
            "returns processing time exceeds policy on Mondays",
        ],
        "observations": [
            "exception clusters align with apartment density",
            "dispatch reuses old plans for 11 minutes after change",
            "pickers walk an extra 60 meters per route",
            "the afternoon spike misses cross-dock by 15 minutes",
            "the slow carrier handles 38 percent of priority volume",
            "Monday returns are 2.4 times average daily volume",
        ],
        "attempts": [
            "we re-routed apartment-dense areas through micro-fulfillment",
            "we shortened the dispatch refresh interval",
            "we re-laid out high-volume SKUs near the picking station",
            "we shifted cross-dock cutoff time earlier",
            "we rebalanced carrier allocation by parcel size",
            "we staffed up returns for Monday morning",
        ],
        "constraints": [
            "carrier contracts cannot change mid-quarter",
            "warehouse layout must respect safety regulations",
            "labor scheduling must follow union agreements",
            "we cannot raise shipping prices this season",
            "package handling must remain auditable end to end",
            "any new automation must be tested off-hours",
        ],
        "decisions": [
            "we will pilot drone-assisted scans in two warehouses",
            "we will reorder SKU adjacency based on co-pick frequency",
            "we will introduce a returns triage rack",
            "we will publish a carrier scorecard internally",
            "we will partition routing by service tier",
            "we will track exception heatmaps weekly",
        ],
        "questions": [
            "is dynamic pricing acceptable for premium delivery",
            "should we offer in-store returns for online orders",
            "do we need a real-time exception notification",
            "is autonomous delivery ready for our suburban routes",
            "should we centralize returns processing",
            "is route optimization SaaS worth the integration cost",
        ],
    },
    "cybersecurity_audit": {
        "subjects": [
            "the access control audit", "the patch management cadence",
            "the dependency vulnerability scan", "the data classification policy",
            "the incident response runbook", "the vendor security review",
        ],
        "issues": [
            "two privileged groups exceed least-privilege guidelines",
            "patch lag for high severity items is over 14 days",
            "vulnerable dependencies persist across services",
            "data classification on shared drives is inconsistent",
            "runbooks are out of date for cloud-native services",
            "vendor reviews vary in rigor across categories",
        ],
        "observations": [
            "92 percent of issues come from three legacy services",
            "patch lag spikes during the holiday months",
            "the same CVE re-appears via transitive dependencies",
            "5 percent of files have no classification label",
            "runbook last-updated dates are older than 90 days",
            "vendor reviews skip vendor-on-vendor risk",
        ],
        "attempts": [
            "we automated quarterly access reviews",
            "we set patch SLAs by severity tier",
            "we adopted SCA gating in CI for high severity items",
            "we ran a scripted classification sweep",
            "we updated runbooks for the top ten services",
            "we standardized vendor questionnaires",
        ],
        "constraints": [
            "we cannot block deploys for low severity items",
            "patches require change approval per environment",
            "data classification cannot rely on manual labeling alone",
            "incident drills must include leadership availability",
            "vendor risk teams have limited bandwidth",
            "auditors require evidence retention for one year",
        ],
        "decisions": [
            "we will deprecate two legacy services this quarter",
            "we will shorten the patch SLA for high severity items",
            "we will gate releases on SCA findings",
            "we will pilot ML-driven classification on shared drives",
            "we will run a quarterly tabletop exercise",
            "we will maintain a centralized vendor registry",
        ],
        "questions": [
            "do we need a privileged access management refresh",
            "should we publish a CVE bulletin internally",
            "is zero-trust architecture practical for our stack",
            "how do we evaluate vendor on-vendor risk",
            "is data labeling worth crowd sourcing internally",
            "should we adopt SBOM as a contract requirement",
        ],
    },
    "financial_fraud_detection": {
        "subjects": [
            "the rule engine", "the anomaly detector",
            "the chargeback workflow", "the model retraining pipeline",
            "the case investigator queue", "the alert tuning policy",
        ],
        "issues": [
            "the false positive rate climbed above the cap",
            "rule conflicts produce duplicate alerts",
            "chargeback throughput exceeds investigator capacity",
            "retraining drifts behind the latest fraud patterns",
            "the case backlog stretches longer than 72 hours",
            "alert tuning lacks per-merchant granularity",
        ],
        "observations": [
            "FPR rose to 3.4 percent against a 2 percent cap",
            "two rules duplicate alerts for high value transactions",
            "investigator queue depth is 1.8 times prior month",
            "the retraining cadence is biweekly but events are weekly",
            "case duration P75 grew from 28 to 41 hours",
            "tuning is global instead of merchant-specific",
        ],
        "attempts": [
            "we calibrated rule thresholds against recent labels",
            "we deduplicated alerts with a consolidated severity",
            "we shifted high-value cases to senior investigators",
            "we shortened retraining intervals to weekly",
            "we added case prioritization by exposure value",
            "we partitioned tuning by merchant size cohorts",
        ],
        "constraints": [
            "we cannot auto-decline beyond a tested risk threshold",
            "investigators are bound by regional regulations",
            "model deployments must include shadow validation",
            "any rule change requires risk committee review",
            "merchant communications must remain compliant",
            "data retention is capped at the agreed period",
        ],
        "decisions": [
            "we will adopt segmented thresholds per merchant size",
            "we will publish a case priority playbook",
            "we will move retraining to a streaming approach",
            "we will run shadow models for two weeks before promotion",
            "we will include feedback loops in the rule engine",
            "we will track FPR per merchant cohort",
        ],
        "questions": [
            "should we adopt graph-based features for collusion",
            "is online learning safe for fraud at our scale",
            "do we need a dedicated investigations team for SMBs",
            "how do we evaluate model fairness across cohorts",
            "should we let merchants tune their own thresholds",
            "is a feedback marketplace useful for shared learning",
        ],
    },
    "nlp_system_design": {
        "subjects": [
            "the entity extraction service", "the multilingual classifier",
            "the summarization pipeline", "the embedding index",
            "the prompt template registry", "the evaluation suite",
        ],
        "issues": [
            "entity extraction misses long-tail organization names",
            "the multilingual classifier degrades on low-resource locales",
            "summarization drops critical numeric facts",
            "the embedding index suffers stale entries after schema changes",
            "prompt templates diverge across teams",
            "the evaluation suite under-represents production traffic",
        ],
        "observations": [
            "F1 on long-tail organizations is below 0.5",
            "low-resource locales account for 9 percent of traffic",
            "summary fact recall drops on tables and figures",
            "stale embeddings linger for 72 hours after schema change",
            "five teams use four different system prompts",
            "the evaluation set has 3 percent coverage of production",
        ],
        "attempts": [
            "we added gazetteer features for organization names",
            "we expanded multilingual training with augmented data",
            "we trained a numeric-aware extraction model",
            "we added a TTL on embedded snippets",
            "we centralized prompt templates with metadata",
            "we sampled production traffic into the evaluation set",
        ],
        "constraints": [
            "we cannot exceed our compute budget for retraining",
            "user data cannot leave the regional cluster",
            "we must maintain model parity for legacy clients",
            "translation models must be auditable",
            "any change must include offline benchmarks",
            "human review is required for new locales",
        ],
        "decisions": [
            "we will publish a shared prompt registry",
            "we will adopt rolling embedding refreshes",
            "we will introduce locale-specific evaluation slices",
            "we will run summarization with numeric-aware decoding",
            "we will track entity extraction by category",
            "we will gate releases on a fairness review",
        ],
        "questions": [
            "is retrieval better than fine-tuning for our locales",
            "should we adopt a single embedding model across products",
            "do we need separate evaluation suites for safety",
            "should we share prompts across product groups",
            "is human-in-the-loop necessary for new locales",
            "how do we measure summary faithfulness reliably",
        ],
    },
    "game_engine_optimization": {
        "subjects": [
            "the rendering pipeline", "the asset streaming system",
            "the physics solver", "the AI navigation mesh",
            "the input handling layer", "the audio mixing system",
        ],
        "issues": [
            "frame time fluctuates on dense outdoor scenes",
            "asset streaming hitches during fast travel",
            "the physics solver under-resolves rapid contacts",
            "the navigation mesh fails on dynamic obstacles",
            "input lag exceeds target on certain controllers",
            "audio mixing clips during scripted events",
        ],
        "observations": [
            "GPU bound passes peak in the foliage area",
            "asset hitches correlate with disk seek bursts",
            "physics tunneling appears at small step sizes",
            "nav mesh updates pause AI for 90 milliseconds",
            "input lag rises with battery savings on console",
            "audio clip events cluster around boss intros",
        ],
        "attempts": [
            "we baked LODs more aggressively for dense foliage",
            "we pre-fetched assets along travel corridors",
            "we increased physics substeps for fast objects",
            "we partitioned the nav mesh into chunks",
            "we polled input on a higher priority thread",
            "we side-chained the music when SFX peaks",
        ],
        "constraints": [
            "the engine must hold 60 FPS on baseline hardware",
            "memory budgets must respect platform constraints",
            "physics changes must keep deterministic gameplay",
            "we cannot ship platform-specific shaders",
            "the audio system must support spatialization on all platforms",
            "build sizes must remain within store limits",
        ],
        "decisions": [
            "we will adopt mesh shaders where supported",
            "we will profile asset streaming per disk class",
            "we will introduce continuous collision detection",
            "we will run nav mesh updates incrementally",
            "we will publish per-platform input policies",
            "we will document the audio mix bus topology",
        ],
        "questions": [
            "is GPU driven rendering ready for our targets",
            "should we adopt virtual texturing",
            "is async compute worth the complexity",
            "do we need a profiler integrated into the editor",
            "should physics share threads with audio",
            "is HDR audio mixing worth the bandwidth",
        ],
    },
    "entrepreneurship": {
        "subjects": [
            "the attendance automation startup", "the early customer pipeline",
            "the pricing experiment", "the partner channel",
            "the founding team responsibilities", "the legal entity setup",
        ],
        "issues": [
            "early customer interviews surface privacy concerns",
            "trial-to-paid conversion sits below target",
            "pricing tiers feel arbitrary to prospects",
            "partner conversations stall after the demo",
            "the founder workload is uneven across responsibilities",
            "the entity setup is delaying our first big deal",
        ],
        "observations": [
            "privacy concerns concentrate in larger institutions",
            "trial conversion is 8 percent versus a 12 percent target",
            "prospects map tiers to seats but we sell by usage",
            "partner deals require a stronger business case",
            "two founders own four core areas each",
            "tax registration takes longer than expected",
        ],
        "attempts": [
            "we adopted privacy-by-design messaging in pitch decks",
            "we shortened the trial to encourage decisions",
            "we ran a willingness-to-pay survey",
            "we wrote a partner-ready ROI calculator",
            "we redistributed founder roles by strength",
            "we engaged a legal advisor for entity setup",
        ],
        "constraints": [
            "the fundraising round closes in three months",
            "we cannot promise on-prem deployment yet",
            "we want to keep customer success cost in check",
            "partner deals must respect data residency",
            "founders must avoid burnout this sprint",
            "we cannot commit to a SaaS-only model right now",
        ],
        "decisions": [
            "we will publish a privacy white paper for institutions",
            "we will reduce trial duration to two weeks",
            "we will pilot a flat tier with usage cap",
            "we will sign one strategic partner before quarter end",
            "we will hire a part-time finance lead",
            "we will register entities in two key markets",
        ],
        "questions": [
            "should we open source a privacy-friendly module",
            "is consumption pricing right for our buyers",
            "do we need a research-led design partner",
            "should we hire technical or commercial first",
            "is going public with the roadmap helpful",
            "should we set up an advisor program",
        ],
    },
    "health_lifestyle": {
        "subjects": [
            "my evening routine", "the workout split",
            "the meal plan", "the steps target",
            "the recovery habits", "the screen time policy",
        ],
        "issues": [
            "evenings drift toward late-night work without buffer",
            "the workout split overworks pushing days",
            "meals lack consistent protein on busy days",
            "step counts dip below target during release weeks",
            "recovery feels poor after long meeting blocks",
            "screen time after dinner reduces sleep quality",
        ],
        "observations": [
            "I lose 90 minutes most evenings to email",
            "my push days correlate with shoulder soreness",
            "lunch protein averages 22 grams below my goal",
            "step counts drop 30 percent on release weeks",
            "long meetings correlate with afternoon slump",
            "device usage past 22:30 correlates with wakeful nights",
        ],
        "attempts": [
            "I scheduled a hard stop at 21:00 on weekdays",
            "I rebalanced the split with more pulling days",
            "I prepared protein-forward lunches in batches",
            "I added micro-walks between meeting blocks",
            "I scheduled deload weeks alongside release sprints",
            "I switched to grayscale on devices after 22:00",
        ],
        "constraints": [
            "I cannot give up on early meetings on Mondays",
            "I want changes that fit work-from-home schedule",
            "I do not want to add new supplements",
            "I want to keep weekend social plans",
            "I cannot move workouts before 06:30",
            "I prefer minimal manual food logging",
        ],
        "decisions": [
            "I will keep the 21:00 hard stop on weekdays",
            "I will rotate workout splits monthly",
            "I will keep batch lunch prep on Sundays",
            "I will check step counts each Friday",
            "I will book a quarterly deload week",
            "I will move late screens away from bed",
        ],
        "questions": [
            "is HRV variation a useful daily decision input",
            "should I add a dedicated mobility day",
            "is intermittent fasting worth trying",
            "is pickleball a good cardio substitute",
            "should I track stress along with sleep",
            "is a small standing desk part of the answer",
        ],
    },
}


# ── NOISE CORPUS ─────────────────────────────────────────────────────────────
# Off-topic distractor sentences. Each line is filtered against the episode's
# relevant content with a low TF-IDF cosine similarity threshold so even
# on-domain-sounding noise does not pollute labels.

NOISE_CORPUS: List[str] = [
    "I started a small herb garden on the balcony last weekend.",
    "I am reading a novel about Antarctic explorers.",
    "My family is planning a road trip across the coast in summer.",
    "I cannot find a comfortable mechanical keyboard at this price point.",
    "I am trying a new coffee bean from a local roastery.",
    "I watched a documentary on Renaissance architecture last night.",
    "I joined a weekend bouldering group in the city.",
    "My neighbor adopted a rescue dog with anxiety issues.",
    "I am restoring a vintage wooden bookshelf this month.",
    "The bakery on the corner started a sourdough subscription.",
    "I am sketching daily portraits to practice anatomy.",
    "My favorite tea shop discontinued an oolong I liked.",
    "I tried a new pasta recipe with anchovies and capers.",
    "I dropped pottery class because the studio raised fees.",
    "I bought a hammock for the small terrace.",
    "I am planning a long-distance bike ride next month.",
    "I caught the seasonal cold last Tuesday.",
    "The local cinema reopened after renovations.",
    "I finally finished a 1000-piece coastal puzzle.",
    "I picked up running shoes with a wider toe box.",
    "I switched to an aluminum kettle for the campsite.",
    "I am building a small wooden bird feeder for the garden.",
    "I tried a sunrise yoga class at the park.",
    "My niece started learning to play violin.",
    "I am testing a new pour-over technique with a longer bloom.",
    "I bought a thicker yoga mat for cold mornings.",
    "I tried a rural farmers market this weekend.",
    "The community theatre is staging an old comedy next month.",
    "I am replanting basil in larger pots after the heatwave.",
    "I joined a book club focused on translated fiction.",
    "I bought leather shoe wax to refresh old boots.",
    "I started keeping a paper journal in the evenings.",
    "I am learning origami cranes for an event.",
    "I made banana bread with walnuts and dark chocolate.",
    "My friend recommended a hot sauce from a small producer.",
    "I switched my morning music to acoustic instrumental playlists.",
    "I am organizing a hiking trip with college friends.",
    "I am doodling small landscapes during breaks.",
    "I bought new linen sheets for the warmer months.",
    "I tried a sweet potato chili recipe over the weekend.",
    "I am preparing decorations for my niece's birthday.",
    "I borrowed a board game from a colleague.",
    "I am planting tomatoes in the new raised bed.",
    "I went to a botanical garden with my partner.",
    "I tried a calligraphy pen with a flexible nib.",
    "I started learning ukulele chords on Sundays.",
    "I bought a film camera at a flea market.",
    "I joined an evening pottery class.",
    "I made a pesto with toasted almonds instead of pine nuts.",
    "I am refurbishing a metal lamp from a thrift store.",
    "I tried homemade kombucha with rosemary.",
    "I picked up a watercolor set with travel pans.",
    "I joined an outdoor swimming group on Tuesdays.",
    "I read three short stories from a Polish author.",
    "I am organizing a kitchen pantry by category.",
    "I bought a new electric grinder for spices.",
    "I am planning to bake macarons for a friend's wedding.",
    "I am rewatching a favorite chef's travel series.",
    "I am crafting a leather wallet from a kit.",
    "I am decorating my desk with succulents.",
    "I tried sourdough discard pancakes for breakfast.",
    "I am trying a longer pour-over recipe with cooler water.",
    "I joined a small gardening community online.",
    "I am painting a sunset over a lake on canvas.",
    "I tried a citrus salad with fennel and pistachios.",
    "I bought new winter gloves for the trail.",
    "I went thrifting and found a wool blazer.",
    "I am ordering bulk herbs for the year ahead.",
    "I started a habit tracker on a paper journal.",
    "I joined a Sunday cycling club for casual rides.",
    "My cat figured out how to open the bedroom door.",
    "I am repainting an old wooden chair in pastel colors.",
    "I tried a new ramen spot near the train station.",
    "I started learning a few phrases in Italian.",
    "I am preparing a gift basket for a friend moving abroad.",
    "I am decluttering the wardrobe by season.",
    "I am planning a summer beach picnic.",
    "I tried a citrus marinade for grilled vegetables.",
    "I am setting up a small outdoor lantern display.",
    "I am volunteering at a weekend library event.",
    "I made a new playlist for foggy mornings.",
    "I caught up with an old college friend over brunch.",
    "My partner started baking sourdough bread.",
    "I tried a homemade vanilla extract recipe.",
    "I am refurbishing an old camera lens housing.",
    "I am cleaning the bicycle drivetrain with citrus degreaser.",
    "I am switching to ceramic coffee filters.",
    "I tried a pumpkin curry recipe with toasted seeds.",
    "I am shopping for a new dining table.",
    "I am rotating houseplants between rooms by light.",
    "I am organizing a board game evening with neighbors.",
    "I tried a smoked tea blend in the afternoon.",
    "I am baking a small chocolate tart for the weekend.",
    "I am framing a couple of vintage maps for the hallway.",
    "I joined a chess club at the community center.",
    "I am cleaning out the garage on Saturday mornings.",
    "I bought a small handheld vacuum for the car.",
    "I am preparing a cold brew batch for the week.",
    "I am trying a new cycling helmet with better airflow.",
    "I tried a spicy mango salsa over grilled fish.",
    "I am exploring stamp collecting from a friend's hobby.",
    "I am making a felt ornament for the holidays.",
    "I am brewing a darker coffee roast for cold mornings.",
    "I tried a savory oatmeal with poached egg and chili oil.",
    "I joined a Sunday morning farmers market run.",
    "I am refilling a glass jar of dried oregano from the garden.",
    "I bought thicker socks for the autumn rain.",
    "I tried a passionfruit ice cream from a local shop.",
    "I am planning a quiet dinner for my parents.",
    "I am teaching my younger cousin to bake brownies.",
    "I bought a wooden chess board for the living room.",
    "I am shopping for a new humidifier for the bedroom.",
    "I tried a low-sugar tiramisu using mascarpone alternatives.",
    "I am keeping a watercolor sketchbook for travel.",
    "I am going to a friend's housewarming next Saturday.",
    "I bought a new pair of trail running shorts.",
    "I tried roasted carrots with tahini and pomegranate.",
    "I am redoing the kitchen tile grout this weekend.",
    "I tried a soft cheese pairing flight at a wine bar.",
    "I am trying a new espresso blend with a fruity finish.",
    "I am buying small frames for a photo wall.",
    "I joined a beach volleyball pickup group.",
    "I tried a tomato gazpacho with cucumber pearls.",
    "I am shopping for tea cups with a soft glaze.",
    "I am binge-listening to an audiobook trilogy.",
    "I am planning a slow weekend drive to a hill town.",
    "I bought waterproof shoe covers for the rainy commute.",
    "I tried a vegan sushi platter at a new place downtown.",
    "I am rearranging my bookshelf by mood.",
]


# ── TF-IDF cosine helpers (no sklearn dependency) ────────────────────────────

_TOK = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOK.findall(text.lower())


def _build_idf(docs: List[str]) -> Tuple[Dict[str, float], float]:
    df: Counter = Counter()
    for doc in docs:
        for tok in set(_tokenize(doc)):
            df[tok] += 1
    n = max(1, len(docs))
    idf = {t: math.log((n + 1.0) / (c + 1.0)) + 1.0 for t, c in df.items()}
    default = math.log((n + 1.0) / 1.0) + 1.0
    return idf, default


def _tfidf(text: str, idf: Dict[str, float], default: float) -> Dict[str, float]:
    toks = _tokenize(text)
    if not toks:
        return {}
    tf = Counter(toks)
    total = float(sum(tf.values()))
    return {t: (c / total) * idf.get(t, default) for t, c in tf.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a).intersection(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na <= 0 or nb <= 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


# ── RELEVANT MESSAGE GENERATION ──────────────────────────────────────────────

def _maybe_capitalize(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


_PATTERNS = [
    "{subjects} {issues}.",
    "{subjects} {issues}, and {observations}.",
    "After {attempts}, {observations}.",
    "We need {constraints}, but {issues}.",
    "{decisions} so we can address {issues}.",
    "{questions}?",
    "{observations}; {decisions}.",
    "{subjects} {issues} because {observations}.",
    "{decisions} after we noted that {observations}.",
    "We tried that {attempts}, however {observations}.",
]


def _compose_message(rng: random.Random, blueprint: Dict[str, List[str]]) -> str:
    pattern = rng.choice(_PATTERNS)
    fragments = {slot: rng.choice(values) for slot, values in blueprint.items()}
    text = pattern.format(**fragments)
    return _maybe_capitalize(text)


def _episode_relevant_messages(
    rng: random.Random,
    blueprint: Dict[str, List[str]],
    n: int,
) -> List[str]:
    seen = set()
    out: List[str] = []
    attempts = 0
    while len(out) < n and attempts < n * 6:
        attempts += 1
        msg = _compose_message(rng, blueprint)
        if msg in seen:
            continue
        seen.add(msg)
        out.append(msg)
    return out


# ── EPISODE BUILDER ──────────────────────────────────────────────────────────

def _build_one_episode(
    rng: random.Random,
    blueprint: Dict[str, List[str]],
    target_len: int,
    relevant_frac: float,
    noise_idf: Dict[str, float],
    noise_default: float,
) -> List[Dict[str, object]]:
    n_relevant = max(2, int(round(target_len * relevant_frac)))
    n_noise = max(1, target_len - n_relevant)

    relevant_msgs = _episode_relevant_messages(rng, blueprint, n_relevant)

    relevant_blob = " ".join(relevant_msgs)
    relevant_vec = _tfidf(relevant_blob, noise_idf, noise_default)

    chosen_noise: List[str] = []
    pool = list(NOISE_CORPUS)
    rng.shuffle(pool)
    for cand in pool:
        if len(chosen_noise) >= n_noise:
            break
        cand_vec = _tfidf(cand, noise_idf, noise_default)
        if _cosine(cand_vec, relevant_vec) > NOISE_COSINE_THRESHOLD:
            continue
        chosen_noise.append(cand)

    if len(chosen_noise) < n_noise:
        for cand in pool:
            if len(chosen_noise) >= n_noise:
                break
            if cand not in chosen_noise:
                chosen_noise.append(cand)

    items: List[Dict[str, object]] = (
        [{"text": m, "isRelevant": True} for m in relevant_msgs]
        + [{"text": m, "isRelevant": False} for m in chosen_noise]
    )
    rng.shuffle(items)
    return items


def main() -> None:
    rng = random.Random(SEED)
    noise_idf, noise_default = _build_idf(NOISE_CORPUS)

    out_episodes: List[Dict[str, object]] = []
    next_id = 1
    for domain, blueprint in DOMAIN_BLUEPRINTS.items():
        difficulty = DIFFICULTY_BY_DOMAIN.get(domain, "medium")
        prof = DIFFICULTY_PROFILE[difficulty]
        for _ in range(NUM_EPISODES_PER_DOMAIN):
            target_len = rng.randint(TARGET_LEN_MIN, TARGET_LEN_MAX)
            relevant_frac = rng.uniform(prof["relevant_frac_min"], prof["relevant_frac_max"])
            if MAX_RELEVANT_PER_EPISODE > 0:
                cap_frac = MAX_RELEVANT_PER_EPISODE / max(1, target_len)
                relevant_frac = min(relevant_frac, cap_frac)
            messages = _build_one_episode(
                rng=rng,
                blueprint=blueprint,
                target_len=target_len,
                relevant_frac=relevant_frac,
                noise_idf=noise_idf,
                noise_default=noise_default,
            )
            out_episodes.append(
                {
                    "episode_id": next_id,
                    "conversation_domain": domain,
                    "difficulty": difficulty,
                    "string_relevant_messages": messages,
                }
            )
            next_id += 1

    OUTPUT_PATH.write_text(
        json.dumps(out_episodes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    n_msgs = sum(len(ep["string_relevant_messages"]) for ep in out_episodes)
    n_relev = sum(
        1
        for ep in out_episodes
        for m in ep["string_relevant_messages"]
        if m["isRelevant"]
    )
    print(f"[OK] wrote {len(out_episodes)} episodes -> {OUTPUT_PATH}")
    print(f"     total messages: {n_msgs}")
    print(f"     relevant: {n_relev}/{n_msgs} = {n_relev/max(1,n_msgs):.2f}")
    print(f"     domains: {len(DOMAIN_BLUEPRINTS)}")


if __name__ == "__main__":
    main()
