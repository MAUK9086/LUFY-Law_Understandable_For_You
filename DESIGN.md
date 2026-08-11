# LUFY — Design Notes

The engineering reasoning behind LUFY: why the system is built this way, what the
naive version got wrong, and what is still outstanding. For installation, API
usage, and configuration, see [README.md](README.md).

**Contents**

- [The Problem](#the-problem)
- [Architecture](#architecture)
- [RAG Pipeline Design](#rag-pipeline-design)
- [Why Naive RAG Fails on Legal Documents](#why-naive-rag-fails-on-legal-documents--and-what-i-did-about-it)
- [Tech Stack Rationale](#tech-stack-rationale)
- [Known Gaps and Future Work](#known-gaps-and-future-work)

---

## The Problem

Legal illiteracy is a structural disadvantage. Tenants sign rental agreements without understanding the eviction clause. Freelancers accept contracts with unlimited liability provisions. Employees join companies with non-compete terms that affect their next five years. The cost of a lawyer to explain these documents puts basic legal comprehension out of reach for most people.

LUFY does not replace a lawyer. It gives users enough information to know when they need one, and what to ask.

**Summary** — Extracts the document's purpose, key parties, critical clauses, financial obligations, and practical implications in five labelled sections. The summary is framed through a persona (tenant, employee, freelancer, or general public) so the language focuses on what matters to that reader.

**Risk Analysis** — Classifies every significant clause into one of three tiers: red (unfair or actively harmful to the user), yellow (vague, missing, or worth negotiating), and green (protective or unambiguously fair). Each flag includes the specific clause, a plain-language explanation, and actionable advice. The system always runs risk analysis on the full document, not on a sampled excerpt.

**Question Answering** — Users can interrogate the document directly. The system retrieves only the most relevant passages and grounds the LLM's answer entirely in that retrieved text. It never answers from general training knowledge. If the answer is not in the document, it says so.

**Multilingual Output** — All three features can be delivered in 16 Indian languages. Translation applies to the full output, including the risk flag labels, so a Hindi speaker sees a complete, consistent experience.

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
        |-- POST /api/summarize       full text (single-shot or map-reduce) -> Groq LLM -> structured summary
        |-- POST /api/risk-analysis   full text (single-shot or map-reduce) -> Groq LLM -> JSON risk flags
        |-- POST /api/query           embed query -> hybrid retrieve -> rerank -> Groq LLM -> answer
        +-- GET  /health              service liveness check


Document ingestion pipeline
  Raw bytes (PDF / DOCX / TXT)
    -> PyMuPDF / python-docx / utf-8 decode
    -> clean_text()          (NFC normalize, collapse whitespace)
    -> split_into_chunks()   (paragraph-aware, 800-char windows, 150-char overlap)
    -> SentenceTransformer   (all-MiniLM-L6-v2, 384-dim vectors, CPU)
    -> ChromaDB collection   (in-memory, cosine similarity space, per session)
    -> BM25Okapi index       (in-process, over the same chunks)


Query pipeline (/api/query)
  User question
    -> embed_query()          (same model, single vector)
    -> hybrid first stage     (ChromaDB dense cosine + BM25 lexical, up to 20 each)
    -> Reciprocal Rank Fusion (score = SUM 1/(60+rank), plus section-header boost)
    -> cross-encoder rerank   (ms-marco-MiniLM-L-6-v2 over the fused top-10)
    -> top-4 chunks
    -> Groq llama-3.1-8b      (numbered source injection in prompt)
    -> answer + source excerpts
    -> deep-translator        (optional, paragraph-chunked)
    -> browser


Whole-document pipeline (/api/summarize, /api/risk-analysis)
  get_full_text()
    -> if <= 60,000 chars (summary) / 50,000 chars (risk):
         single Groq call, original path, unchanged
    -> else map-reduce fallback:
         group_chunks_for_budget()        (budget-sized batches)
         -> map: concurrent AsyncGroq calls, semaphore-bounded,
                 retry with backoff on 429
         -> summary: reduce concatenated notes through the five-section prompt
         -> risk:    merge flag lists, de-duplicate by normalised clause text
                     (no reduce call — flags are discrete items, not prose)
```

All state is in-memory. The FastAPI process holds one ChromaDB client, one embedding model (singleton, loaded at startup), and a session dictionary mapping session IDs to their collections. There is no Redis, no PostgreSQL, no file system writes.

---

## RAG Pipeline Design

### Why RAG Instead of Full-Text Prompting

A 20-page contract has roughly 30,000–50,000 characters. That fits in a modern context window, but feeding the whole document on every question is wasteful, and LLMs attend poorly to content buried in the middle of a long prompt (the "lost in the middle" problem). Retrieval-augmented generation fixes this by feeding the model only the 3–4 most relevant passages for a given query, keeping the context tight and the signal-to-noise ratio high.

Note that this argument applies to *queries* only. Summarisation and risk analysis have no query to retrieve against and genuinely need the whole document — they take the separate whole-document path described above. Conflating the two is what caused failure mode 5 below.

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

---

## Why Naive RAG Fails on Legal Documents — and What I Did About It

A first cut of this system used the textbook RAG recipe: paragraph chunks, dense embeddings, top-4 cosine retrieval, and "join the chunks back together" for whole-document tasks. On legal text, each of those defaults has a specific failure mode. Here is what broke and the concrete change that fixed it.

**1. Reconstructing context from overlapping chunks wastes the budget.** Summarisation and risk analysis need the *whole* document, so the original code did `"\n\n".join(chunks)`. But chunks carry 150 characters of overlap each, so a 10,000-character document was reassembled into 12,000+ characters of partially duplicated text — the LLM paid attention budget to the same sentences twice, and the overlap seams read as noise. The fix is trivial once you see it: the cleaned pre-chunk text already exists (`ParsedDocument.raw_text`), so the session now stores it directly and `get_full_text()` returns the original, not a reassembly. (`vector_store.py`)

**2. Dense-only retrieval misses exact legal terms.** Bi-encoder cosine similarity is excellent at *topical* matching but blurs *lexical* precision. A question like "what happens if I break the lease early?" should land on the clause containing "early termination" — but a 384-dim sentence embedding scores a semantically adjacent passage about "notice periods" almost as highly. Legal language is full of terms of art ("indemnify", "severability", a specific defined term) where an exact token match is the strongest possible signal. The fix is a **BM25 lexical index fused with the dense retriever via Reciprocal Rank Fusion** — BM25 rewards the exact-term hit, dense rewards the paraphrase, and RRF combines them without needing their raw scores to be comparable. (`vector_store.py`)

**3. Paragraph chunking discards document structure.** Splitting on `\n\n` treats a two-line "DEFINITIONS" header and a substantive indemnity clause as equal citizens. But a legal document's section structure *is* signal: a query about termination should prefer chunks under the "TERMINATION" heading. So chunking now **detects section headers and tags every chunk with its section**, stored as ChromaDB metadata, and retrieval applies a small boost when the query terms overlap a chunk's section header. (`text_utils.py`, `vector_store.py`)

**4. First-stage retrieval is recall-oriented, not precision-oriented.** HNSW + BM25 are fast approximate retrievers; their job is to *not miss* the answer, not to rank it first. Handing the top-4 of a fuzzy first stage straight to the LLM means the genuinely best clause is often at rank 3 or 4, where "lost in the middle" degrades the answer. The fix is a **cross-encoder reranker**: retrieve a wider net (top-10 fused), then rescore each `(query, chunk)` pair with a model that reads them *together* rather than comparing pre-computed vectors. This is the single change that most improves answer quality, and it is ~15 lines because the bi-encoder infrastructure already exists. (`reranker.py`)

**5. Retrieval only covered the Q&A feature — summarisation and risk analysis were silently truncating long documents at a budget far below what the model can actually handle.** RAG (retrieval + reranking) only applies to `/api/query`, where there's a query to retrieve against. Summarisation and risk analysis need the *whole* document, not the top-k passages for a question — so they were sending the full cleaned text to the LLM in one call, hard-truncated at a fixed character budget (originally 12,000 / 10,000 chars). On a genuinely long contract, a clause past that budget was simply never seen by the model — the exact failure mode this system exists to prevent, just relocated from "buried in a huge prompt" to "cut off entirely."

The first instinct was to reach for map-reduce everywhere, but checking `llama-3.1-8b-instant`'s actual context window first (128K tokens) showed the original budget was using roughly 2% of what the model can take. So the real fix has two parts:

- **Raise the single-shot budget** to something actually sized to the model (60,000 / 50,000 chars — still well under the context window, but comfortably above the 20-page-contract estimate of 30,000–50,000 characters). Most real documents now take the original one-call path, unchanged, with the truncation bug fixed for them at zero added cost.
- **Keep map-reduce as a fallback** for the rare document that's still too long even at that budget, not the default path.

When the fallback does trigger: documents are split into budget-sized batches (`group_chunks_for_budget`), and each batch is independently summarised into terse notes (or risk-analysed with the same JSON contract) in a "map" pass. That map pass runs concurrently, bounded by a semaphore (`_gather_bounded`, capped at `MAP_REDUCE_MAX_CONCURRENCY`) to stay under Groq's free-tier rate limit rather than firing every batch at once, with retry and backoff on 429s (`_call_groq_async`) since concurrent calls are the case actually likely to hit that limit. For summarisation, the concatenated notes are run through the normal five-section prompt as a "reduce" pass; for risk analysis, the per-batch flag lists are merged and de-duplicated by normalised clause text (batches overlap slightly, so a clause can otherwise surface twice) — no reduce LLM call needed, since flags are discrete items rather than prose. (`llm_client.py`: `summarise_document_long`, `analyse_risks_long`)

**The throughline:** naive RAG optimises for the average document, but legal text rewards exact terms, respects strict structure, and punishes wasted context — and "whole-document" tasks need their own long-document strategy sized to what the model can actually do, because retrieval doesn't apply when there's no query to retrieve against. Each stage above trades a little latency for precision or completeness where it matters.

---

## Tech Stack Rationale

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
| Frontend | Vanilla HTML/CSS/JS | No build step, no Node.js dependency, no bundler. The SPA logic is ~500 lines; a framework would add more complexity than it removes |
| Container | Docker python:3.11-slim | Matches Hugging Face Spaces Docker SDK expectation; slim base keeps the image under 2 GB including model weights |

---

## Known Gaps and Future Work

**Blocking I/O on the event loop.** Every route handler is `async def`, but the work inside them is synchronous and blocks the event loop for its full duration:

- All Groq calls on the ordinary paths (`summarise_document`, `analyse_risks`, `answer_query`) use the sync `Groq` client with no `run_in_executor`. Only the map-reduce fallback uses `AsyncGroq`.
- `translate()` is a synchronous `deep-translator` call, invoked from all three async routes.
- `parse_document()` and `embed_chunks()` run inline in the `async def upload_document` handler, so embedding a large document stalls every concurrent request.

Under concurrent users this caps throughput regardless of the map-reduce work. Fixing it properly means moving the whole app onto `AsyncGroq` and pushing the CPU-bound parsing and embedding into a thread pool — a larger, separate change.

**Single-shot budgets are hardcoded.** `_SINGLE_SHOT_CHAR_LIMIT` is a module constant in `app/api/routes/summarize.py` (60,000) and `app/api/routes/risk.py` (50,000), while every other tunable — including the two supporting map-reduce settings — lives in `config.py` as an environment variable. The threshold that decides whether the fallback runs at all should be configurable alongside them.

**Sessions are process-local.** The session dictionary lives in the FastAPI process, so the app cannot be scaled horizontally without sticky sessions or an external store, and a restart drops every active session. This is a deliberate trade for the privacy model, not an oversight — but it is a real ceiling.
