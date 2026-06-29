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

Chunking is also **structure-aware**: while walking paragraphs, lines that look like legal section headers — ALL-CAPS labels, numbered clauses (`1.`, `2.3`), or short colon-terminated labels — are detected and recorded as the "current section" (`_is_section_header` in `text_utils.py`). Each chunk is tagged with the section it belongs to, and that label is stored as ChromaDB metadata. The retriever later uses it to boost chunks whose section header overlaps the query terms.

### Embedding Model Choice

`all-MiniLM-L6-v2` produces 384-dimensional sentence embeddings. It is small enough to run on CPU without perceptible latency (the model is ~23 MB after quantization), semantically accurate for short paragraphs, and has a permissive Apache 2.0 licence. The model is loaded once at server startup via `asynccontextmanager` + `run_in_executor` and cached with `@lru_cache`. Subsequent embedding calls add no model-loading overhead.

The `local_files_only=True` flag prevents the sentence-transformers library from making HuggingFace Hub HEAD requests on every call — a behaviour introduced in version 3.x that added 5–10 seconds of latency even when the model was already cached locally.

### Vector Store Choice

ChromaDB runs fully in-process with no daemon or network dependency. For a stateless, per-session workload (each user gets a collection that is created on upload and discarded on session end), it requires no configuration and adds no operational complexity. The collection is created with `hnsw:space: cosine` so that retrieval scores map directly to semantic similarity (1.0 = identical, 0.0 = orthogonal).

### Retrieval and Answer Grounding

Retrieval is a three-stage pipeline rather than a single cosine-similarity lookup:

1. **Hybrid first-stage retrieval.** Each query is run against two retrievers in parallel — ChromaDB dense (cosine) similarity and a BM25 lexical index built over the same chunks. Each returns up to 20 candidates.
2. **Reciprocal Rank Fusion (RRF).** The two ranked lists are merged with RRF (`score = Σ 1 / (60 + rank)`), so a chunk that ranks highly in *either* retriever surfaces, and chunks ranked highly in *both* win. Chunks whose section header shares keywords with the query receive a small additional boost.
3. **Cross-encoder reranking.** The fused top-10 candidates are rescored by a `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder, which jointly encodes each `(query, chunk)` pair for a much sharper relevance estimate than the first-stage bi-encoder. The reranked top-`RETRIEVAL_TOP_K` (4) chunks are what the LLM sees.

These are injected into the prompt as numbered excerpts:

```
[Source 1] ... text ...
[Source 2] ... text ...
```

The LLM prompt instructs the model to answer only from the numbered sources and to quote directly where possible. If the answer is not in the excerpts, the model is instructed to return a fixed string ("This specific information is not covered in the retrieved sections of the document.") rather than hallucinating.

Source citations are shown to the user below each answer, with sentence-level extraction: the citation display runs a lightweight overlap scorer against the user's query words, selects the 2–3 most relevant sentences from each chunk, and shows those rather than the full 800-character window. This keeps the citation readable without hiding context.

### Why This Design Is Well-Suited to Legal Retrieval

Legal documents have a strong sectional structure. Clauses about rent are grouped together; termination conditions are grouped together. Paragraph-based chunking aligns with this structure. Cosine similarity on sentence embeddings reliably separates "liability" sections from "payment" sections because the vocabulary is sufficiently distinct. The overlap prevents the retriever from missing a clause that references a defined term from the preceding paragraph.

The alternative — a sliding window over raw character positions — produces chunks that can split sentences and mix clauses from unrelated sections, degrading both retrieval precision and the coherence of the LLM's answer.

### Why Naive RAG Fails on Legal Documents — and What I Did About It

A first cut of this system used the textbook RAG recipe: paragraph chunks, dense embeddings, top-4 cosine retrieval, and "join the chunks back together" for whole-document tasks. On legal text, each of those defaults has a specific failure mode. Here is what broke and the concrete change that fixed it.

**1. Reconstructing context from overlapping chunks wastes the budget.** Summarisation and risk analysis need the *whole* document, so the original code did `"\n\n".join(chunks)`. But chunks carry 150 characters of overlap each, so a 10,000-character document was reassembled into 12,000+ characters of partially duplicated text — the LLM paid attention budget to the same sentences twice, and the overlap seams read as noise. The fix is trivial once you see it: the cleaned pre-chunk text already exists (`ParsedDocument.raw_text`), so the session now stores it directly and `get_full_text()` returns the original, not a reassembly. (`vector_store.py`)

**2. Dense-only retrieval misses exact legal terms.** Bi-encoder cosine similarity is excellent at *topical* matching but blurs *lexical* precision. A question like "what happens if I break the lease early?" should land on the clause containing "early termination" — but a 384-dim sentence embedding scores a semantically adjacent passage about "notice periods" almost as highly. Legal language is full of terms of art ("indemnify", "severability", a specific defined term) where an exact token match is the strongest possible signal. The fix is a **BM25 lexical index fused with the dense retriever via Reciprocal Rank Fusion** — BM25 rewards the exact-term hit, dense rewards the paraphrase, and RRF combines them without needing their raw scores to be comparable. (`vector_store.py`)

**3. Paragraph chunking discards document structure.** Splitting on `\n\n` treats a two-line "DEFINITIONS" header and a substantive indemnity clause as equal citizens. But a legal document's section structure *is* signal: a query about termination should prefer chunks under the "TERMINATION" heading. So chunking now **detects section headers and tags every chunk with its section**, stored as ChromaDB metadata, and retrieval applies a small boost when the query terms overlap a chunk's section header. (`text_utils.py`, `vector_store.py`)

**4. First-stage retrieval is recall-oriented, not precision-oriented.** HNSW + BM25 are fast approximate retrievers; their job is to *not miss* the answer, not to rank it first. Handing the top-4 of a fuzzy first stage straight to the LLM means the genuinely best clause is often at rank 3 or 4, where "lost in the middle" degrades the answer. The fix is a **cross-encoder reranker**: retrieve a wider net (top-10 fused), then rescore each `(query, chunk)` pair with a model that reads them *together* rather than comparing pre-computed vectors. This is the single change that most improves answer quality, and it is ~15 lines because the bi-encoder infrastructure already exists. (`reranker.py`)

The throughline: naive RAG optimises for the average document, but legal text rewards exact terms, respects strict structure, and punishes wasted context. Each stage above trades a little latency for precision where it matters.

---

## Tech Stack

| Component | Technology | Reason |
|---|---|---|
| Backend framework | FastAPI | Async-native, automatic OpenAPI docs, Pydantic integration, fast enough for this workload without Starlette complexity |
| ASGI server | Uvicorn | Standard companion for FastAPI; supports lifespan events needed for model pre-warming |
| LLM | Groq API / llama-3.1-8b-instant | Sub-second inference at no cost (free tier); llama-3.1-8b follows structured prompts reliably; avoids running a local LLM that would require GPU |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 | CPU-only, 23 MB, Apache-2.0; 384-dim vectors are accurate for paragraph-length legal text |
| Reranker | sentence-transformers cross-encoder ms-marco-MiniLM-L-6-v2 | CPU-friendly cross-encoder (~80 MB); jointly encodes (query, chunk) for a precision second stage over the fused candidates |
| Lexical retrieval | rank-bm25 (BM25Okapi) | In-process BM25 index fused with dense retrieval via RRF; catches exact legal terms that embeddings blur |
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
    reranker.py           — CrossEncoder singleton, rerank() precision stage
    vector_store.py       — ChromaDB + BM25 sessions, hybrid retrieve() w/ RRF, get_full_text()
    llm_client.py         — Groq API calls, structured prompts per endpoint
    risk_analyzer.py      — RiskFlag/RiskReport Pydantic models, parse_risk_response()
    translator.py         — deep-translator wrapper, paragraph-chunked for long texts
  utils/
    text_utils.py         — clean_text(), truncate_to_token_budget(), structure-aware chunking

tests/                    — pytest suite (text utils, risk parsing, RRF, vector store)

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

On first start, the embedding model (~23 MB) and the cross-encoder reranker (~80 MB) are downloaded from HuggingFace Hub and cached locally. Subsequent starts load from cache in under one second. The server pre-warms both models in the lifespan hook so the first user request is not blocked by model loading.

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
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | cross-encoder reranker model name |
| `MAX_CHUNK_SIZE` | `800` | Maximum characters per document chunk |
| `CHUNK_OVERLAP` | `150` | Character overlap between adjacent chunks |
| `RERANK_TOP_N` | `10` | Candidates passed from hybrid retrieval into the reranker |
| `RETRIEVAL_TOP_K` | `4` | Final chunks kept after reranking and sent to the LLM |
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
