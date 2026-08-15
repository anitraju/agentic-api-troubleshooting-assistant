# Agentic API Troubleshooting Assistant Using RAG

An evidence-grounded API troubleshooting assistant built with Python, LangGraph, RAG,
Sentence Transformers, Chroma, LangChain, and OpenAI.

## Current status

**Commit 8 — LangGraph agent orchestration**

The project now contains a stateful four-node troubleshooting workflow:

```text
START
  |
  v
classify
  |
  v
retrieve
  |
  +---- no evidence ----------------------+
  |                                       |
  v                                       v
diagnose                              verify
  |                                       |
  v                                       |
verify <---- grounding retry --------------+
  |
  v
 END
```

## Agent nodes

### 1. classify

The classifier converts the user's query into a structured `IssueClassification` with an issue
category and only the service, endpoint, HTTP method, and status code that are explicit in the
query.

### 2. retrieve

Classification-derived metadata is merged with any explicit CLI filters. Caller-provided
filters take precedence. The existing semantic retriever then searches the persistent Chroma
knowledge index.

### 3. diagnose

The existing grounded generator creates a structured diagnosis from retrieved evidence. Causes,
diagnostic steps, and remediation steps must cite retrieved evidence ranks.

### 4. verify

The verifier checks every generated evidence rank. If grounding fails, LangGraph conditionally
routes back to `diagnose` with corrective feedback. Retries are bounded by:

```text
AGENT_MAX_GENERATION_ATTEMPTS=2
```

If all attempts fail, the graph returns a safe low-confidence response with sources but no
unsupported diagnosis.

If retrieval returns no evidence, generation is skipped entirely.

## Run the LangGraph agent

Install/update dependencies:

```bash
pip install -e ".[dev]"
```

Build the index if needed:

```bash
python -m scripts.build_index
```

Run:

```bash
python -m scripts.agent_troubleshoot \
  "Why am I getting 401 when calling the order API?"
```

A more explicit failure:

```bash
python -m scripts.agent_troubleshoot \
  "POST /api/v1/orders returns 503. What should I investigate?"
```

Explicit filters are still supported:

```bash
python -m scripts.agent_troubleshoot \
  "Order creation is unavailable" \
  --service order-service \
  --endpoint /api/v1/orders \
  --http-method POST \
  --status-code 503
```

The CLI prints a compact trace followed by the grounded diagnosis.

## Tests

```bash
ruff check .
pytest
```

Commit 8 tests cover classification normalization, metadata-filter merging, the four-node happy
path, no-evidence routing, grounding retries, retry exhaustion, and caller-filter precedence.

The agent tests use fakes and do not call OpenAI or Hugging Face.

## Commit boundary

Commit 8 is intentionally limited to four-node LangGraph orchestration. It does not add a UI,
production-system tools, long-term memory, a checkpointer, or automatic remediation.

## Next step

The next project stage can focus on evaluation and example scenarios: repeatable troubleshooting
cases, retrieval/answer quality checks, and documented agent runs.

## Commit message

```text
add LangGraph troubleshooting agent orchestration
```
