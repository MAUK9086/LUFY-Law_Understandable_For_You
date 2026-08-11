# LUFY — Law Understandable For You

Legal documents are written for courts, not for the people who sign them. LUFY bridges that gap. Upload any legal document — a rental agreement, employment contract, NDA, or court order — and receive a plain-language summary, a three-tier risk assessment, and a grounded question-answering interface, all in your preferred Indian language.

> **📐 [Design Notes → DESIGN.md](DESIGN.md)** — architecture, the RAG pipeline, why naive RAG fails on legal text and what was done about it, tech-stack rationale, and known gaps.

---

## Features

- **Summary** — the document's purpose, parties, critical clauses, financial obligations, and practical implications, in five labelled sections, framed for a chosen persona (tenant, employee, freelancer, or general public).
- **Risk Analysis** — every significant clause sorted into red (harmful), yellow (vague or negotiable), or green (protective) flags, each with a plain-language explanation and actionable advice. Always run over the full document, never a sampled excerpt.
- **Question Answering** — ask the document directly. Answers are grounded strictly in retrieved passages with sentence-level citations; if the answer is not in the document, it says so.
- **Multilingual Output** — all three features available in 16 Indian languages, including the risk flag labels.

LUFY does not replace a lawyer. It gives users enough information to know when they need one, and what to ask.

### Privacy

No document is stored on disk. Sessions live in in-process memory and are discarded when the server restarts. There is no database, no user accounts, and no logging of document content. The only external call that sees document text is the Groq API request, covered by Groq's privacy policy. Translation goes through the Google Translate API via `deep-translator`, which receives text chunks but no session identifiers.

---

## Quickstart

**Prerequisites:** Python 3.11 or later, and a free Groq API key from [console.groq.com](https://console.groq.com).

```bash
git clone https://github.com/MAUK9086/LUFY-Law_Understandable_For_You.git
cd LUFY-Law_Understandable_For_You
pip install -r requirements.txt
cp .env.example .env
# Set GROQ_API_KEY in .env
uvicorn app.main:app --reload --port 7860
```

| | |
|---|---|
| Landing page | `http://localhost:7860` |
| Application | `http://localhost:7860/app.html` |
| API docs | `http://localhost:7860/docs` |
| Health check | `http://localhost:7860/health` |

On first start the embedding model (~23 MB) and cross-encoder reranker (~80 MB) are downloaded from HuggingFace Hub and cached. Subsequent starts load from cache in under a second; both models are pre-warmed in the lifespan hook so the first request is not blocked on model loading.

### Docker

```bash
docker build -t lufy-v2 .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key_here lufy-v2
```

The Dockerfile pre-bakes the model weights at build time, so cold starts are deterministic.

### Tests

```bash
pytest
```

---

## API Reference

All endpoints accept and return JSON, except `/api/upload` which takes `multipart/form-data`.

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/health` | — | `{status, service}` |
| POST | `/api/upload` | multipart `file` | `{session_id, filename, page_count, char_count, chunk_count}` |
| POST | `/api/demo` | — | same as upload |
| POST | `/api/summarize` | `{session_id, persona, language}` | `{summary, language}` |
| POST | `/api/risk-analysis` | `{session_id, persona, language}` | `{red_flags, yellow_flags, green_flags, section_labels}` |
| POST | `/api/query` | `{session_id, query, persona, language}` | `{answer, sources}` |

`session_id` is returned by `/api/upload` or `/api/demo` and must be passed to all subsequent calls. Sessions are in-memory only and are lost on server restart.

---

## Configuration

All settings are read from environment variables or a `.env` file.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *required* | Groq API key from console.groq.com |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model identifier |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model name |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker model name |
| `MAX_CHUNK_SIZE` | `800` | Maximum characters per document chunk |
| `CHUNK_OVERLAP` | `150` | Character overlap between adjacent chunks |
| `RERANK_TOP_N` | `10` | Candidates passed from hybrid retrieval into the reranker |
| `RETRIEVAL_TOP_K` | `4` | Final chunks kept after reranking and sent to the LLM |
| `MAP_REDUCE_BATCH_CHARS` | `6000` | Characters per map-step call when summarisation or risk analysis falls back to map-reduce |
| `MAP_REDUCE_MAX_CONCURRENCY` | `3` | Maximum concurrent map-step calls, kept under Groq's free-tier rate limit |
| `DEBUG` | `false` | Enable debug-level logging |

---

## Project Structure

```
app/
  config.py               — pydantic-settings BaseSettings singleton (@lru_cache)
  main.py                 — FastAPI factory, lifespan pre-warm, static mount
  api/
    schemas.py            — all HTTP boundary models (Pydantic v2)
    routes/
      health.py           — GET /health
      document.py         — POST /api/upload, POST /api/demo
      summarize.py        — POST /api/summarize
      risk.py             — POST /api/risk-analysis
      query.py            — POST /api/query (RAG entrypoint)
  core/
    document_processor.py — PDF/DOCX/TXT parsing, ParsedDocument dataclass
    embedder.py           — SentenceTransformer singleton, embed_chunks, embed_query
    reranker.py           — CrossEncoder singleton, rerank() precision stage
    vector_store.py       — ChromaDB + BM25 sessions, hybrid retrieve() w/ RRF, get_full_text()
    llm_client.py         — Groq calls, prompt building, map-reduce long-document path
    risk_analyzer.py      — RiskFlag/RiskReport Pydantic models, parse_risk_response()
    translator.py         — deep-translator wrapper, paragraph-chunked for long texts
  utils/
    text_utils.py         — clean_text(), structure-aware chunking, budget grouping

tests/                    — pytest suite (text utils, risk parsing, RRF, vector store, LLM client)

frontend/
  index.html              — landing page
  app.html                — single-page application (two-column layout)
  assets/                 — fonts, images, CSS for landing page
  static/
    css/app.css           — design system via CSS variables, no framework
    js/app.js             — state machine (IDLE/UPLOADING/UPLOADED/ANALYSING/READY)

sample_docs/              — bundled legal notice, referral agreement, and judgment for demo mode
Dockerfile
.env.example
requirements.txt
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
