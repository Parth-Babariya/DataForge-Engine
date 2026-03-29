# DataForge Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

> An end-to-end dataset engineering platform that generates, validates, and serves synthetic Q&A data using LLMs.

## Overview
DataForge Engine is a full-stack pipeline for synthetic Q&A dataset generation with quality checks and export-ready storage.

### Highlights
- LLM-powered domain-specific Q&A generation
- Validation and deduplication pipeline
- JSONL persistence (dataset-friendly format)
- FastAPI backend + Streamlit frontend
- Pluggable LLM client abstraction

## Architecture
```text
Streamlit UI -> FastAPI API -> Services (Generator, Validator, Store) -> JSONL Datasets
```

## Tech Stack
- FastAPI
- Streamlit
- OpenAI API
- Pydantic
- Pandas
- Pytest
- Docker / Docker Compose

## Project Structure
```text
dataforge_engine/
+-- src/
|   +-- api/
|   +-- interfaces/
|   +-- services/
|   +-- ui/
+-- tests/
+-- Dockerfile
+-- docker-compose.yml
+-- requirements.txt
+-- README.md
```

## Quick Start
### Prerequisites
- Python 3.9+ or Docker
- OpenAI API key

### Option A: Docker
```bash
git clone https://github.com/Parth-Babariya/DataForge-Engine.git
cd dataforge-engine
cp .env.example .env
# add OPENAI_API_KEY in .env
docker-compose up --build
```

### Option B: Local
```bash
git clone https://github.com/Parth-Babariya/DataForge-Engine.git
cd dataforge-engine
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add OPENAI_API_KEY in .env
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

In another terminal:
```bash
streamlit run src/ui/app.py
```

## API
### Generate dataset
```bash
curl -X POST "http://localhost:8000/api/v1/generate" \\
  -H "Content-Type: application/json" \\
  -d '{"domain":"science","num_samples":5}'
```

### Health check
```bash
curl http://localhost:8000/health
```

## Testing
```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term
```

## Configuration
Key environment variables:
- `OPENAI_API_KEY` (required)
- `LLM_MODEL` (default: `gpt-3.5-turbo`)
- `LLM_TEMPERATURE` (default: `0.7`)
- `DATASET_BASE_DIR` (default: `datasets`)
- `VALIDATOR_MIN_LENGTH` (default: `10`)
- `LOG_LEVEL` (default: `INFO`)

## License
MIT
