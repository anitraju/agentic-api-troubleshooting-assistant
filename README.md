# Agentic API Troubleshooting Assistant Using RAG

A portfolio project for building an AI assistant that investigates API failures using
retrieval-augmented generation (RAG), structured troubleshooting tools, and an agentic
workflow.

## Current status

**Commit 7 — Baseline grounded RAG troubleshooting pipeline**

The project can now retrieve relevant technical evidence and turn it into a structured,
evidence-grounded troubleshooting diagnosis using an OpenAI chat model through LangChain.

```text
User troubleshooting query
          ↓
SemanticRetriever
          ↓
Ranked RetrievalResult evidence
          ↓
Grounded prompt context
          ↓
ChatOpenAI structured output
          ↓
Grounding validation
          ↓
TroubleshootingAnswer + citations
```

## Grounding design

The LLM does not receive an unrestricted system description. It receives only the user query
and the ranked evidence returned by Commit 6.

Every generated likely cause, diagnostic step, and remediation step must provide one or more
`evidence_ranks`. The pipeline validates those ranks against the evidence actually retrieved.
A generated statement that cites a non-existent rank raises `UngroundedGenerationError`.

If retrieval returns no evidence, the pipeline does not call the LLM. It returns a low-confidence
response that explicitly says a grounded diagnosis cannot be made.

## Structured output

The generation schema contains:

```text
summary
likely_causes[]
  text
  evidence_ranks[]
diagnostic_steps[]
  text
  evidence_ranks[]
remediation_steps[]
  text
  evidence_ranks[]
confidence
limitations[]
```

The final `TroubleshootingAnswer` also carries the exact `RetrievalResult` objects used to
produce the diagnosis, allowing citations to be rendered deterministically.

## LLM integration

Commit 7 introduces `langchain-openai` and uses `ChatOpenAI.with_structured_output(...)` with
a Pydantic schema. Tests use fakes and never call the OpenAI API.

The OpenAI HTTP client uses the operating-system trust store through `truststore`, matching the
certificate strategy already used for Hugging Face model downloads on managed machines.

## Configuration

Add these values to your local `.env`:

```text
OPENAI_API_KEY=<your API key>
OPENAI_MODEL=gpt-5-nano
LLM_TIMEOUT_SECONDS=60
LLM_MAX_RETRIES=2
```

Do not commit your real API key. `.env` remains local.

## Install dependencies

```bash
pip install -e ".[dev]"
```

## Build or verify the knowledge index

```bash
python -m scripts.build_index
```

The current sample knowledge base should still contain 61 chunks.

## Run the baseline RAG assistant

```bash
python -m scripts.troubleshoot \
  "Why am I getting 401 when calling the order API?"
```

You can reuse the Commit 6 metadata filters:

```bash
python -m scripts.troubleshoot \
  "Order creation is returning service unavailable" \
  --service order-service \
  --http-method POST \
  --status-code 503
```

The rendered answer contains a diagnosis, confidence level, evidence-backed causes, diagnostic
steps, remediation, limitations when needed, and the retrieved sources.

## Tests

```bash
ruff check .
pytest
```

Commit 7 tests do not require an OpenAI API key. They cover structured answer validation,
evidence-context formatting, citation rendering, retrieve-then-generate orchestration,
empty-retrieval fallback behavior, filter forwarding, and rejection of ungrounded evidence
references.

## Files added or changed

```text
app/
├── config.py
└── rag/
    ├── __init__.py
    ├── answers.py
    ├── generation.py
    └── pipeline.py

scripts/
└── troubleshoot.py

tests/
├── test_answers.py
├── test_generation.py
└── test_pipeline.py

.env.example
pyproject.toml
README.md
```

## Architecture boundary

Commit 7 is a deterministic **retrieve → generate → validate** RAG pipeline. It intentionally
does not yet make autonomous decisions, branch between workflows, call diagnostic tools, retry
failed reasoning, or maintain agent state.

## Next commit

Commit 8 will introduce LangGraph orchestration with explicit troubleshooting state and nodes
for classification, retrieval, diagnosis, and verification.

## Commit message

```text
feat: add grounded RAG troubleshooting pipeline
```
