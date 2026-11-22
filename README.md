# TradeIntel AI

TradeIntel AI is an end-to-end **RAG-based commodity intelligence API** built to demonstrate how AI can support document-heavy commodity trading workflows. It ingests trade documents, retrieves relevant evidence for a question, and optionally uses an OpenAI model to generate a grounded answer with source excerpts.

## Why this project
Commodity teams work with market briefs, supplier offers, specifications, logistics notes and internal documents. TradeIntel AI provides a practical backend for searching this information through natural-language questions while keeping answers tied to retrieved source text.

## Features
- FastAPI REST backend with automatic Swagger documentation
- TXT/PDF document ingestion
- Chunking and TF-IDF retrieval with cosine similarity
- Retrieval-Augmented Generation (RAG) workflow
- Optional OpenAI LLM integration via environment variable
- Source excerpts and retrieval scores for traceability
- Retrieval-only fallback when no API key is configured
- Input validation, health endpoint and API tests
- Docker support and PostgreSQL-ready configuration

## Architecture
`Documents -> Text extraction -> Chunking -> Retrieval index -> Top-k context -> LLM (optional) -> Grounded answer + sources`

## Tech stack
Python, FastAPI, Pydantic, scikit-learn, OpenAI API, PyPDF, Pytest, Docker. A PostgreSQL connection string is included as the next persistence layer for production expansion.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open `http://127.0.0.1:8000/docs` for the interactive API UI.

The project works without an OpenAI key in retrieval-only mode. To enable generated RAG answers:
```bash
cp .env.example .env
export OPENAI_API_KEY="your-key"
```

## Example query
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What factors matter in iron ore procurement?","top_k":3}'
```

## API endpoints
- `GET /` - project status
- `GET /health` - health and indexed chunk count
- `POST /documents` - upload TXT/PDF documents
- `POST /query` - retrieve context and generate a grounded response

## Responsible AI considerations
The system exposes source excerpts instead of presenting generated text as unsupported truth. A production version should add authentication, document-level access controls, stronger evaluation datasets, embedding/vector storage, prompt-injection defenses, PII handling and monitoring for latency/cost.

## Interview talking points
1. Why RAG is preferable to relying on an LLM's internal knowledge for changing commodity information.
2. How chunk size and retrieval quality affect answer accuracy.
3. How you would replace TF-IDF with embeddings + pgvector/FAISS for scale.
4. How to measure groundedness, retrieval recall, latency and cost.
5. How FastAPI, PostgreSQL and Docker would fit a production deployment.

## Disclaimer
The included commodity documents are demonstration data created for this project and are not live market advice.
