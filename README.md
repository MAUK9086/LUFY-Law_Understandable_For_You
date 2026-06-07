# LUFY — Law Understandable For You

LUFY simplifies complex legal documents into plain-language summaries, risk assessments, and direct answers — all in your preferred Indian language. Upload a PDF, DOCX, or TXT and get instant insights powered by a full RAG pipeline.

---

## Architecture

```
Browser
  │
  ├── GET  /              → frontend/index.html  (landing page)
  ├── GET  /app.html      → frontend/app.html    (application UI)
  │
  └── FastAPI (uvicorn)
        ├── POST /api/upload          → parse → embed → ChromaDB session
        ├── POST /api/demo            → load sample_docs/legal_judgement.txt
        ├── POST /api/summarize       → full text → Groq LLM → plain summary
        ├── POST /api/risk-analysis   → full text → Groq LLM → JSON flags
        ├── POST /api/query           → embed query → retrieve → Groq LLM answer
        └── GET  /health              → {"status":"ok"}

Core pipeline
  Document bytes
    → PyMuPDF / python-docx / utf-8 decode
    → clean_text + split_into_chunks
    → sentence-transformers (all-MiniLM-L6-v2, CPU)
    → ChromaDB in-memory collection (cosine space)
    → Groq API (llama-3.1-8b-instant)
    → deep-translator (Google backend) — optional
```

---

## Key Features

- **Plain-language summaries** — persona-aware (tenant / employee / freelancer / general)
- **Three-tier risk analysis** — red / yellow / green flags with clause, explanation, and advice
- **RAG question answering** — answers grounded in retrieved document excerpts with citations
- **16 Indian languages** — Hindi, Gujarati, Marathi, Tamil, Telugu, Bengali, Kannada, Malayalam, Punjabi, Odia, Assamese, Urdu, Sanskrit, Sindhi, Kashmiri
- **Format support** — PDF, DOCX, TXT (up to 10 MB)
- **No persistence** — all data lives in-memory per session; nothing is stored on disk

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| LLM | Groq API — llama-3.1-8b-instant |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local CPU) |
| Vector DB | ChromaDB in-memory |
| PDF parsing | PyMuPDF (fitz) |
| DOCX parsing | python-docx |
| Translation | deep-translator (GoogleTranslator) |
| Frontend | Vanilla HTML / CSS / JS — no build step |
| Container | Docker (python:3.11-slim) |
| Hosting target | Hugging Face Spaces Docker SDK (port 7860) |

---

## Local Development

### Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com/)

### Steps

```bash
# 1. Clone
git clone https://github.com/your-username/LUFY.git
cd LUFY

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_here

# 4. Start the server
uvicorn app.main:app --reload --port 7860
```

Open `http://localhost:7860` — landing page loads.
Open `http://localhost:7860/app.html` — application UI.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model ID |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model name |
| `MAX_CHUNK_SIZE` | `800` | Max characters per document chunk |
| `CHUNK_OVERLAP` | `150` | Overlap characters between chunks |
| `RETRIEVAL_TOP_K` | `4` | Chunks retrieved per query |
| `DEBUG` | `false` | Enable debug logging |

---

## Docker

```bash
# Build (pre-bakes the embedding model weights into the image)
docker build -t lufy-v2 .

# Run
docker run -p 7860:7860 -e GROQ_API_KEY=your_key_here lufy-v2
```

---

## API Reference

| Method | Path | Body / Params | Returns |
|---|---|---|---|
| GET | `/health` | — | `{status, service}` |
| POST | `/api/upload` | `file` (multipart) | `UploadResponse` |
| POST | `/api/demo` | — | `UploadResponse` |
| POST | `/api/summarize` | `{session_id, persona, language}` | `{summary, language}` |
| POST | `/api/risk-analysis` | `{session_id, persona, language}` | `{red_flags, yellow_flags, green_flags}` |
| POST | `/api/query` | `{session_id, query, persona, language}` | `{answer, sources}` |

Interactive docs: `http://localhost:7860/docs`

---

## Project Structure

```
app/
  config.py               — pydantic-settings singleton
  main.py                 — FastAPI factory
  api/
    schemas.py            — request/response models
    routes/
      health.py
      document.py         — /upload, /demo
      summarize.py        — /summarize
      risk.py             — /risk-analysis
      query.py            — /query (RAG)
  core/
    document_processor.py — PDF/DOCX/TXT parsing + chunking
    embedder.py           — sentence-transformers wrapper
    vector_store.py       — ChromaDB session management
    llm_client.py         — Groq API calls
    risk_analyzer.py      — Pydantic risk models + validation
    translator.py         — deep-translator wrapper
  utils/
    text_utils.py         — clean_text, truncate, split_into_chunks

frontend/
  index.html              — landing page
  app.html                — single-page application
  assets/                 — CSS, JS, fonts, images (landing page)
  static/
    css/app.css
    js/app.js

sample_docs/              — bundled sample documents
Dockerfile
.env.example
requirements.txt
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
