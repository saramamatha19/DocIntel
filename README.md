# DocIntel

**AI Document Intelligence and RAG Assistant** — ask questions across your company's documents and get answers grounded in exact, cited passages, not hallucinated guesses.

DocIntel isn't a single embedding-lookup-and-prompt demo. It's a multi-stage retrieval pipeline — hybrid search, cross-encoder reranking, diversity-aware selection, conversational query rewriting, and confidence-gated guardrails — wrapped in a document-management UI that understands PDFs, Word docs, plain text, Markdown, and live webpages.

---

## Why this is more than a tutorial RAG app

Most RAG walkthroughs stop at "embed the query, grab the top-k chunks, ask an LLM." DocIntel treats that as the *baseline*, not the destination:

| Typical tutorial RAG | DocIntel |
|---|---|
| Single vector search | **Hybrid search** — vector similarity + BM25 keyword search, merged with Reciprocal Rank Fusion |
| Top-k by raw similarity | **Cross-encoder reranking** — a second model re-scores candidates by actually reading the query and passage together |
| Whatever top-k comes out | **MMR diversity selection** — actively avoids handing the LLM five near-duplicate chunks that all repeat the same clause |
| Follow-up questions silently break | **Conversational query rewriting** — "what about their pricing?" is resolved into a standalone question *before* retrieval runs |
| Answers with no source | **Inline `[N]` citations** with hover popups showing the exact quoted passage |
| No sense of "was this a good answer?" | **Calibrated confidence scoring**, empirically tuned against real relevant/irrelevant queries — with a pre-LLM-call guardrail that skips the (paid) generation step entirely when retrieval confidence is too low |
| One document pile | **Multi-company organization** + a **Compare** feature for both company-vs-company Q&A and document-vs-document "what changed" diffing |

Every one of these was implemented against a real, reproduced failure case, not a hypothetical one — and every limitation still on the table is documented, not hidden (see [Engineering notes](#engineering-notes) below).

---

## How a question actually gets answered

```mermaid
flowchart TD
    Q[User question] --> RW{Is this a follow-up?}
    RW -- yes --> RWQ[Rewrite into a standalone question]
    RW -- no --> S
    RWQ --> S[Search query]
    S --> V[Vector search, top 20]
    S --> K[BM25 keyword search, top 20]
    V --> F[Reciprocal Rank Fusion]
    K --> F
    F --> RR[Cross-encoder reranks all 20 candidates]
    RR --> MMR[MMR picks the final 5, balancing relevance vs redundancy]
    MMR --> G{Confidence check}
    G -- too low --> N[Skip the LLM call, return a clear no-answer]
    G -- good enough --> L[LLM writes a cited answer from those 5 chunks only]
    L --> R[Frontend renders answer with hover citations and a confidence meter]
```

And how a document gets in, in the first place:

```mermaid
flowchart LR
    U[Upload: PDF, DOCX, TXT, MD, or a URL] --> E[Extract text, tables, and OCR any images]
    E --> C[Chunk]
    C --> P[Redact emails and passwords]
    P --> AI[Auto-classify into a category and auto-summarize]
    AI --> EM[Embed locally with MiniLM-L6-v2]
    EM --> DB[(ChromaDB, persistent)]
```

---

## Features

**Advanced retrieval**
- Hybrid search (vector + BM25, fused with RRF)
- Cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`, runs locally, zero API cost)
- MMR diversity selection to reduce redundant sources
- Conversational query rewriting for pronoun/context-dependent follow-ups

**Trust and safety**
- Two-layer confidence guardrail: skips the LLM call entirely on clearly unanswerable questions, and cleans up the display if the LLM refuses anyway
- Calibrated 0–100% confidence meter (good / warn / critical), tuned against real queries — not a made-up formula
- Inline `[N]` citations with hover popups showing the exact quoted source text
- PII redaction (emails and labeled passwords) applied at chunking time, before anything is stored

**Document understanding**
- Multi-format ingestion: PDF, DOCX, TXT, Markdown, and live webpages
- Table extraction and image OCR inside PDFs (not just plain text)
- Automatic 9-category classification (HR, Legal & Compliance, Security & IT, Engineering & Product, Finance, Sales & Marketing, Operations & Facilities, Executive & Strategy, Other) and one-line auto-summary per document
- Duplicate detection by content hash (files) or URL (webpages)

**Multi-company workspace**
- Every document tagged by company at upload time
- Document management UI: grouped by company, with category badges, chunk counts, and delete
- **Compare** mode, two ways:
  - *Company vs. company* — ask one question, get a side-by-side answer with independent confidence scores for each side
  - *Document vs. document* — pick two specific documents and get an "Added / Removed / Changed" diff

**Visibility**
- A live, streaming "what is it doing" trace for every question (retrieval → confidence → generation), not a generic spinner
- Structured logging (`docintel.log`) of every ask/compare call, including confidence, source count, timing, and rewritten queries

---

## Tech stack

| Layer | Choice |
|---|---|
| Backend API | FastAPI + Uvicorn, streaming responses (NDJSON) |
| Orchestration | LangChain (LCEL chains) |
| LLM | OpenAI `gpt-4o-mini` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2`, local, normalized vectors |
| Vector store | ChromaDB, persistent, local |
| Keyword search | `rank-bm25` (BM25Okapi) |
| Reranking | `sentence-transformers` `CrossEncoder` |
| Frontend | Streamlit |
| Document parsing | PyMuPDF (PDF + OCR via `pytesseract`), `python-docx` |

Every dependency choice here was deliberate about weight: BM25 was picked over `langchain_community`'s retriever wrapper to keep the mechanism transparent and the dependency tiny; the cross-encoder reranker piggybacks on `sentence-transformers`, which was already installed for embeddings, so reranking added **zero** new dependencies.

---

## Getting started

### Prerequisites
- Python 3.12
- An OpenAI API key
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed system-wide (used for scanning images embedded in PDFs) — e.g. `brew install tesseract` on macOS

### Setup

```bash
# 1. Clone and enter the project
git clone <this-repo>
cd DocIntel

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
echo "OPENAI_API_KEY=sk-..." > .env
```

### Run it

Two terminals, both from the project root:

```bash
# Terminal 1 — backend (http://127.0.0.1:8000)
cd backend
uvicorn main:app --reload
```

```bash
# Terminal 2 — frontend
cd frontend
streamlit run frontend.py
```

Streamlit will open the app in your browser.

---

## Using it

1. **Upload** — go to the Chat tab's sidebar, set a company name, and drop in a PDF, DOCX, TXT, or MD file (or paste a URL). You'll see an auto-generated category and summary immediately.
2. **Ask** — type a question. Watch the live status trace show retrieval, confidence, and generation as they actually happen. Ask a pronoun-based follow-up ("what about their...?") and see it get resolved before retrieval runs.
3. **Inspect** — hover any `[N]` citation to see the exact source passage it came from; check the confidence meter to gauge how grounded the answer is.
4. **Manage** — the Documents tab lists everything indexed, grouped by company, with one-click delete.
5. **Compare** — the Compare tab lets you either ask one question across two companies, or diff two specific document versions directly.

---

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/documents/upload` | Upload a file (PDF/DOCX/TXT/MD) for one company |
| `POST` | `/documents/upload-url` | Ingest a webpage by URL |
| `GET` | `/documents` | List all indexed documents |
| `DELETE` | `/documents` | Delete a document and its chunks |
| `POST` | `/ask` | Ask a question (streaming NDJSON response) |
| `POST` | `/compare` | Compare two companies on one question (streaming) |
| `POST` | `/compare-documents` | Diff two specific documents (streaming) |

---

## Project structure

```
backend/
  main.py               # FastAPI routes — thin HTTP layer only
  chroma_db.py           # Owns all ChromaDB-specific logic
  retriever.py            # Hybrid search, reranking, MMR, confidence scoring
  call_llm.py             # All LLM prompts and chains (answer, compare, classify, summarize, query rewriting)
  document_ingestion.py   # PDF/DOCX/TXT extraction, table parsing, image OCR
  chunking.py              # Chunk splitting + PII redaction hook
  embedding.py             # Local embedding model
  pii_redaction.py         # Regex-based email/password redaction
frontend/
  frontend.py             # Streamlit UI — Chat, Documents, Compare tabs
```

---

## Engineering notes

A few honest notes, because a project that pretends to have no edges is less trustworthy than one that names them:

- **Hybrid search and reranking measurably help, but don't fix everything.** One known case — a webpage whose content is short marketing copy sitting alongside long, keyword-dense legal PDFs — still loses out on both vector similarity *and* BM25 keyword density, because the PDFs simply repeat the relevant words more often. Reranking can only reorder whatever hybrid search's fusion stage already shortlisted; it can't rescue a candidate that never made the shortlist. This is a corpus-composition limitation, not a bug, and it's a genuinely interesting one to talk through.
- **Guardrails were built as two layers on purpose**: a pre-LLM-call check (skip generation entirely when retrieval confidence is critical, saving real API cost) and a post-call cleanup (in case the LLM refuses anyway on a borderline "warn"-band retrieval). Confidence isn't guessed at from the answer — it's computed from retrieval before the LLM is ever called.
- **Dependency footprint was a first-class design constraint**, not an afterthought — a full NER model was skipped for PII detection (regex covers emails/passwords reliably; names would need real training data to do responsibly), and the full `ragas` evaluation package was avoided because it pulls in a heavy, version-conflicting dependency tree for a project this size.
- **Confidence scoring was recalibrated after adding reranking and MMR, not assumed to still be correct.** The original floor/ceiling were calibrated against plain vector search, where the reported score is always the single closest chunk in the whole corpus. Reranking and MMR can deliberately keep a chunk that *isn't* the closest embedding match — that's the point of reranking — so that assumption no longer strictly held. A batch of 20 real relevant/irrelevant test queries run through the new pipeline confirmed a measurable drift (one borderline query flipped from "good" to "warn" band), and the floor/ceiling were re-derived from the actual observed score distribution rather than left unchanged on faith. Irrelevant-query confidence dropped from as high as 20% to consistently 0-1% as a result.

## Roadmap

- RAGAS-style automated evaluation (via a lightweight custom harness, pending a decision on the full package's dependency weight)
- Dedicated multi-document reasoning (scope still being decided — `/ask` and Compare may already cover most of it)
- AI-suggested follow-up questions
- An analytics dashboard over asked questions and confidence trends
