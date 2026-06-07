# LUFY — Law Understandable For You

Legal documents are written for courts, not for the people who sign them. LUFY bridges that gap. Upload any legal document — a rental agreement, employment contract, NDA, or court order — and receive a plain-language summary, a three-tier risk assessment, and a grounded question-answering interface, all in your preferred Indian language.

---

## Business Logic

### The Problem

Legal illiteracy is a structural disadvantage. Tenants sign rental agreements without understanding the eviction clause. Freelancers accept contracts with unlimited liability provisions. Employees join companies with non-compete terms that affect their next five years. The cost of a lawyer to explain these documents puts basic legal comprehension out of reach for most people.

### What LUFY Does

LUFY does not replace a lawyer. It gives users enough information to know when they need one, and what to ask.

**Summary** — Extracts the document's purpose, key parties, critical clauses, financial obligations, and practical implications in five labelled sections. The summary is framed through a persona (tenant, employee, freelancer, or general public) so the language focuses on what matters to that reader.

**Risk Analysis** — Classifies every significant clause into one of three tiers: red (unfair or actively harmful to the user), yellow (vague, missing, or worth negotiating), and green (protective or unambiguously fair). Each flag includes the specific clause, a plain-language explanation, and actionable advice. The system always runs risk analysis on the full document, not on a sampled excerpt.

**Question Answering** — Users can interrogate the document directly. The system retrieves only the most relevant passages and grounds the LLM's answer entirely in that retrieved text. It never answers from general training knowledge. If the answer is not in the document, it says so.

**Multilingual Output** — All three features can be delivered in 16 Indian languages. Translation applies to the full output, including the risk flag labels, so a Hindi speaker sees a complete, consistent experience.

### Privacy Model

No document is stored on disk. Sessions live in-process memory and are discarded when the server restarts. There is no database, no user accounts, and no logging of document content. The only external call that sees document text is the Groq API request, which is covered by Groq's privacy policy. Translation is handled by the Google Translate API via `deep-translator`, which receives text chunks but no session identifiers.

---

## Architecture

```
Browser (Vanilla JS SPA)
  |
  |-- GET  /                 frontend/index.html   (landing page)
  |-- GET  /app.html         frontend/app.html     (application)
  |-- GET  /static/*         CSS, JS               (no build step)
  |
  +-- FastAPI (Uvicorn, port 7860)
        |
        |-- POST /api/upload          document bytes -> parse -> embed -> ChromaDB
        |-- POST /api/demo            load sample_docs/ -> same pipeline
        |-- POST /api/summarize       full text -> Groq LLM -> structured summary
        |-- POST /api/risk-analysis   full text -> Groq LLM -> JSON risk flags
        |-- POST /api/query           embed query -> retrieve -> Groq LLM -> answer
        +-- GET  /health              service liveness check


Document ingestion pipeline
  Raw bytes (PDF / DOCX / TXT)
    -> PyMuPDF / python-docx / utf-8 decode
    -> clean_text()          (NFC normalize, collapse whitespace)
    -> split_into_chunks()   (paragraph-aware, 800-char windows, 150-char overlap)
    -> SentenceTransformer   (all-MiniLM-L6-v2, 384-dim vectors, CPU)
    -> ChromaDB collection   (in-memory, cosine similarity space, per session)


Query pipeline
  User question
    -> embed_query()         (same model, single vector)
    -> ChromaDB .query()     (HNSW ANN search, top-4 chunks)
    -> Groq llama-3.1-8b    (numbered source injection in prompt)
    -> answer + source excerpts
    -> deep-translator       (optional, paragraph-chunked)
    -> browser
```

All state is in-memory. The FastAPI process holds one ChromaDB client, one embedding model (singleton, loaded at startup), and a session dictionary mapping session IDs to their collections. There is no Redis, no PostgreSQL, no file system writes.

---

## RAG Pipeline Design

### Why RAG Instead of Full-Text Prompting

A 20-page contract has roughly 30,000–50,000 characters. At GPT-4-class context sizes this would fit in one call, but it is inefficient and expensive. More importantly, LLMs with large contexts attend poorly to content buried in the middle of a long prompt (the so-called "lost in the middle" problem). Retrieval-augmented generation fixes this by feeding the model only the 3–4 most relevant passages for a given query, keeping the context tight and the signal-to-noise ratio high.

### Chunking Strategy

Documents are split on paragraph boundaries first (`\n\n`). Paragraphs that exceed the 800-character window are further split on sentence boundaries (`. `) rather than character position. This preserves semantic units: a clause is not cut mid-sentence. Each chunk carries 150 characters of overlap from the preceding chunk, preventing the retriever from missing an answer that spans two adjacent sections.

### Embedding Model Choice

`all-MiniLM-L6-v2` produces 384-dimensional sentence embeddings. It is small enough to run on CPU without perceptible latency (the model is ~23 MB after quantization), semantically accurate for short paragraphs, and has a permissive Apache 2.0 licence. The model is loaded once at server startup via `asynccontextmanager` + `run_in_executor` and cached with `@lru_cache`. Subsequent embedding calls add no model-loading overhead.

The `local_files_only=True` flag prevents the sentence-transformers library from making HuggingFace Hub HEAD requests on every call — a behaviour introduced in version 3.x that added 5–10 seconds of latency even when the model was already cached locally.

### Vector Store Choice

ChromaDB runs fully in-process with no daemon or network dependency. For a stateless, per-session workload (each user gets a collection that is created on upload and discarded on session end), it requires no configuration and adds no operational complexity. The collection is created with `hnsw:space: cosine` so that retrieval scores map directly to semantic similarity (1.0 = identical, 0.0 = orthogonal).

### Retrieval and Answer Grounding

For each query, the system retrieves the top 4 chunks ranked by cosine similarity. These are injected into the prompt as numbered excerpts:

```
[Source 1] ... text ...
[Source 2] ... text ...
```

The LLM prompt instructs the model to answer only from the numbered sources and to quote directly where possible. If the answer is not in the excerpts, the model is instructed to return a fixed string ("This specific information is not covered in the retrieved sections of the document.") rather than hallucinating.

Source citations are shown to the user below each answer, with sentence-level extraction: the citation display runs a lightweight overlap scorer against the user's query words, selects the 2–3 most relevant sentences from each chunk, and shows those rather than the full 800-character window. This keeps the citation readable without hiding context.

### Why This Design Is Well-Suited to Legal Retrieval

Legal documents have a strong sectional structure. Clauses about rent are grouped together; termination conditions are grouped together. Paragraph-based chunking aligns with this structure. Cosine similarity on sentence embeddings reliably separates "liability" sections from "payment" sections because the vocabulary is sufficiently distinct. The overlap prevents the retriever from missing a clause that references a defined term from the preceding paragraph.

The alternative — a sliding window over raw character positions — produces chunks that can split sentences and mix clauses from unrelated sections, degrading both retrieval precision and the coherence of the LLM's answer.

---

## Tech Stack

| Component | Technology | Reason |
|---|---|---|
| Backend framework | FastAPI | Async-native, automatic OpenAPI docs, Pydantic integration, fast enough for this workload without Starlette complexity |
| ASGI server | Uvicorn | Standard companion for FastAPI; supports lifespan events needed for model pre-warming |
| LLM | Groq API / llama-3.1-8b-instant | Sub-second inference at no cost (free tier); llama-3.1-8b follows structured prompts reliably; avoids running a local LLM that would require GPU |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 | CPU-only, 23 MB, Apache-2.0; 384-dim vectors are accurate for paragraph-length legal text |
| Vector store | ChromaDB in-memory | Zero-config, in-process, per-session collections; no daemon required; HNSW index with cosine space |
| PDF parsing | PyMuPDF (fitz) | Fastest Python PDF library; handles scanned-layout PDFs better than pdfminer; BSD licence |
| DOCX parsing | python-docx | De facto standard; extracts paragraph structure rather than raw text |
| Translation | deep-translator (GoogleTranslator) | Google Translate has the best coverage for low-resource Indian languages; deep-translator wraps it without an API key |
| Data validation | Pydantic v2 | Enforced at every HTTP boundary; model_validator used for post-parse invariant checks |
| Settings | pydantic-settings | Typed environment variable loading with `.env` file support and no boilerplate |
| Frontend | Vanilla HTML/CSS/JS | No build step, no Node.js dependency, no bundler. The SPA logic is 400 lines; a framework would add more complexity than it removes |
| Container | Docker python:3.11-slim | Matches Hugging Face Spaces Docker SDK expectation; slim base keeps the image under 2 GB including model weights |

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
    vector_store.py       — ChromaDB session lifecycle, retrieve(), get_full_text()
    llm_client.py         — Groq API calls, structured prompts per endpoint
    risk_analyzer.py      — RiskFlag/RiskReport Pydantic models, parse_risk_response()
    translator.py         — deep-translator wrapper, paragraph-chunked for long texts
  utils/
    text_utils.py         — clean_text(), truncate_to_token_budget(), split_into_chunks()

frontend/
  index.html              — landing page
  app.html                — single-page application (two-column layout)
  assets/                 — fonts, images, CSS for landing page
  static/
    css/app.css           — design system via CSS variables, no framework
    js/app.js             — state machine (IDLE/UPLOADING/UPLOADED/ANALYSING/READY)

sample_docs/              — bundled sample legal judgment for demo mode
Dockerfile
.env.example
requirements.txt
```

---

## Local Development

**Prerequisites:** Python 3.11 or later, a free Groq API key from console.groq.com.

```bash
git clone https://github.com/your-username/LUFY.git
cd LUFY
pip install -r requirements.txt
cp .env.example .env
# Set GROQ_API_KEY in .env
uvicorn app.main:app --reload --port 7860
```

On first start, the embedding model is downloaded from HuggingFace Hub (~23 MB) and cached locally. Subsequent starts load from cache in under one second. The server pre-warms the model in the lifespan hook so the first user request is not blocked by model loading.

- Landing page: `http://localhost:7860`
- Application: `http://localhost:7860/app.html`
- API docs: `http://localhost:7860/docs`
- Health check: `http://localhost:7860/health`

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | required | Groq API key from console.groq.com |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model identifier |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model name |
| `MAX_CHUNK_SIZE` | `800` | Maximum characters per document chunk |
| `CHUNK_OVERLAP` | `150` | Character overlap between adjacent chunks |
| `RETRIEVAL_TOP_K` | `4` | Number of chunks retrieved per query |
| `DEBUG` | `false` | Enable debug-level logging |

---

## Docker

```bash
docker build -t lufy-v2 .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key_here lufy-v2
```

The Dockerfile pre-bakes the embedding model weights during the image build step. This avoids the HuggingFace Hub download on container startup and makes cold starts deterministic.

---

## API Reference

All endpoints accept and return JSON. The `/api/upload` endpoint accepts `multipart/form-data`.

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

## License

MIT License — see [LICENSE](LICENSE) for details.
