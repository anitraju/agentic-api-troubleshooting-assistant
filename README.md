# Agentic API Troubleshooting Assistant Using RAG

An evidence-grounded API troubleshooting assistant built with Python, LangGraph, RAG,
Sentence Transformers, Chroma, LangChain, and OpenAI.

## Current status

**Commit 9 — Repeatable agent evaluation framework**

Commit 9 adds a curated evaluation dataset and deterministic code-based evaluators so changes
to prompts, retrieval, classification, or graph routing can be measured instead of judged only
from individual demos.

The evaluation flow is:

```text
Curated scenario
      |
      v
LangGraph troubleshooting agent
      |
      v
Final graph state
      |
      +--> classification evaluator
      +--> explicit-signal evaluator
      +--> retrieval-source evaluator
      +--> answer-term recall
      +--> grounding evaluator
      |
      v
Case score + pass/fail
      |
      v
Aggregate evaluation report
```

## Evaluation dimensions

Each case can define reference expectations for:

- issue category;
- status code;
- HTTP method;
- endpoint;
- service;
- one or more expected source hints;
- answer terms expected to appear in the grounded response.

The evaluator also verifies that every evidence citation in the final answer points to a rank
that was actually retrieved.

Category accuracy and grounding are hard requirements. The remaining scores are averaged with
those dimensions and compared with each case's `minimum_score`.

## Curated dataset

The initial dataset is:

```text
data/evaluation/scenarios.json
```

It includes representative cases for:

- 401 authentication;
- 403 authorization / insufficient scope;
- 400 request validation;
- 404 order not found;
- 429 rate limiting;
- 503 service availability.

The cases are intentionally small and human-readable so they can evolve with the knowledge
base.

## Unit tests

Evaluation unit tests are deterministic and do not call OpenAI or Hugging Face:

```bash
ruff check .
pytest
```

They cover:

- evaluation-model normalization;
- aggregate report metrics;
- classification scoring;
- retrieval-source scoring;
- answer-term recall;
- evidence-grounding checks;
- runner exception isolation;
- dataset validation.

## Run a small live evaluation first

A live evaluation uses the real LangGraph agent and therefore consumes OpenAI API credits.

Start with two cases:

```bash
python -m scripts.evaluate_agent --limit 2
```

Example summary:

```text
## Agent evaluation

Cases: 2
Passed: 2
Pass rate: 100.0%
Average score: 0.950

[PASS] auth-401-invalid-token score=1.000
[PASS] authz-403-scope score=0.900
```

Because model output is non-deterministic, the exact score can vary between runs.

## Run one named case

```bash
python -m scripts.evaluate_agent \
  --case-id availability-503-create-order
```

Multiple `--case-id` options can be supplied.

## Run the full curated dataset

```bash
python -m scripts.evaluate_agent
```

## Save a JSON report

```bash
python -m scripts.evaluate_agent \
  --output reports/commit9-evaluation.json
```

The report contains per-case scores, failures, actual classification values, retrieved sources,
confidence, and generation-attempt counts.

## Optional CI-style failure

By default, the evaluator prints failures without returning a failing shell status. To make the
command exit with status `1` when any scenario fails:

```bash
python -m scripts.evaluate_agent --fail-on-regression
```

This is useful later in CI once thresholds are stable.

## Architecture through Commit 9

```text
Knowledge documents
      |
      v
Chunking + metadata
      |
      v
Embeddings + Chroma
      |
      v
Semantic retrieval
      |
      v
Grounded RAG generation
      |
      v
LangGraph
classify -> retrieve -> diagnose -> verify
      |
      v
Commit 9 evaluation harness
dataset -> scorers -> report
```

## Commit boundary

Commit 9 does not add:

- an LLM-as-judge evaluator;
- LangSmith-hosted datasets;
- production tracing dashboards;
- a web UI;
- production API/log tools;
- automatic remediation.

The deterministic local eval layer is intentionally established first. A future enhancement can
mirror the same cases into LangSmith for experiment tracking and LLM-as-judge evaluation.

## Commit message

```text
add repeatable agent evaluation framework
```
