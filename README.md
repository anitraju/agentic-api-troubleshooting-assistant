# Agentic API Troubleshooting Assistant Using RAG

A portfolio project for building an AI assistant that investigates API failures using
retrieval-augmented generation (RAG), structured troubleshooting tools, and an agentic
workflow.

## Current status

**Commit 1 — Project initialization**

This commit establishes the repository structure, Python packaging, environment-based
configuration, console logging, and development tooling.

Later commits will add:

- realistic API troubleshooting data
- document ingestion and metadata-aware chunking
- embeddings and vector retrieval
- a baseline RAG troubleshooting pipeline
- OpenAPI inspection
- log and incident analysis tools
- a LangGraph-based agent
- evidence synthesis and confidence scoring
- FastAPI endpoints
- evaluation, observability, Docker, and CI

## Requirements

- Python 3.11 or newer
- `pip`
- Git

## Setup

Clone the repository and move into it:

```bash
git clone <your-repository-url>
cd agentic-api-troubleshooter
```

Create and activate a virtual environment:

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the project and development dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create your local environment file:

### macOS / Linux

```bash
cp .env.example .env
```

### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

## Run

```bash
python -m app.main
```

Expected output will be similar to:

```text
INFO | __main__ | Starting Agentic API Troubleshooting Assistant
INFO | __main__ | Environment: development
INFO | __main__ | Project initialization completed successfully.
```

Timestamps will also appear in the log output.

## Code quality

Run the linter:

```bash
ruff check .
```

Run tests:

```bash
pytest
```

At this stage there are no functional tests yet; those will be added as the project gains
behavior worth testing.

## Project structure

```text
agentic-api-troubleshooter/
├── app/
│   ├── __init__.py
│   ├── config.py
│   └── main.py
├── data/
│   ├── docs/
│   ├── incidents/
│   ├── logs/
│   ├── openapi/
│   └── runbooks/
├── scripts/
├── tests/
│   └── __init__.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

The empty `data/` and `scripts/` directories are placeholders for upcoming commits.

## Commit message

```text
chore: initialize API troubleshooting assistant project
```
