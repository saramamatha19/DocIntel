import hashlib
import json
import logging
import time
from pathlib import Path

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from document_ingestion import extract_webpage
from chunking import chunk_document
from chroma_db import (
    ingest_document,
    store_chunks,
    list_documents,
    delete_document,
    find_duplicate,
    classify_and_summarize,
    get_document_chunks,
)

from retriever import retrieve_documents, compute_confidence
from call_llm import (
    call_llm,
    compare_answer,
    compare_document_versions,
    NO_ANSWER_MESSAGE,
)


# ============================================================
# 0. Logging configuration
#
# Writes to a persistent file (docintel.log, in whatever
# directory the server is run from) as well as the console, so
# there's a real record of what happened after the terminal
# that ran the server is gone — not just print() statements
# visible only while it's running.
# ============================================================

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("docintel.log"),
        logging.StreamHandler(),
    ],
)

# The root level above is WARNING so third-party libraries
# (httpx logs every single OpenAI API call and every test
# request at INFO level) don't flood the file — only this
# app's own logger is turned up to actually be verbose.
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("docintel")
logger.setLevel(logging.INFO)


# ============================================================
# 1. Create FastAPI application
# ============================================================

app = FastAPI(
    title="DocIntel",
    description="AI Document Intelligence and RAG Assistant",
    version="1.0.0",
)


# ============================================================
# 2. Upload directory
# ============================================================

UPLOAD_DIR = Path("uploads")

UPLOAD_DIR.mkdir(
    exist_ok=True
)

ALLOWED_UPLOAD_EXTENSIONS = (
    ".pdf",
    ".txt",
    ".docx",
    ".md",
)


# ============================================================
# 3. Request models
# ============================================================

class ChatTurn(BaseModel):
    question: str
    answer: str


class QuestionRequest(BaseModel):
    question: str
    chat_history: list[ChatTurn] = []


class URLRequest(BaseModel):
    url: str
    company: str


class DocumentDeleteRequest(BaseModel):
    document_name: str


class CompareRequest(BaseModel):
    question: str
    company_a: str
    company_b: str


class CompareDocumentsRequest(BaseModel):
    document_a: str
    document_b: str
    focus: str = ""


# ============================================================
# 4. Home endpoint
# ============================================================

@app.get("/")
def home():

    return {
        "message": "DocIntel Backend is Running!!!"
    }


# ============================================================
# 5. Upload a document (PDF, TXT, DOCX, or MD)
# ============================================================

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    company: str = Form(...),
):

    # Check file type
    if not file.filename.lower().endswith(ALLOWED_UPLOAD_EXTENSIONS):

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. Allowed: "
                f"{', '.join(ALLOWED_UPLOAD_EXTENSIONS)}"
            )
        )

    document_type = Path(file.filename).suffix.lower().lstrip(".")

    # Read bytes once, so they can be hashed and then saved
    file_bytes = await file.read()

    source_hash = hashlib.sha256(file_bytes).hexdigest()

    # Check whether this exact file is already indexed
    duplicate_name = find_duplicate(
        source_hash=source_hash
    )

    if duplicate_name:

        logger.info(
            f"Upload duplicate skipped: {file.filename} "
            f"(already indexed as '{duplicate_name}')"
        )

        return {

            "filename": file.filename,

            "duplicate": True,

            "message": (
                f"Already indexed as '{duplicate_name}', "
                "skipped."
            ),

            "document_type": document_type,

            "chunks_stored": 0,
        }

    # Create file path
    file_path = UPLOAD_DIR / file.filename

    # Save uploaded file
    with file_path.open("wb") as buffer:

        buffer.write(file_bytes)

    # Ingest document into ChromaDB
    result = ingest_document(
        str(file_path),
        source_hash=source_hash,
        company=company,
    )

    logger.info(
        f"Upload indexed: {file.filename} "
        f"({result['chunks_stored']} chunks, "
        f"category={result['category']}, "
        f"company={company})"
    )

    return {

        "filename": file.filename,

        "duplicate": False,

        "message": (
            "Document uploaded and "
            "indexed successfully"
        ),

        "document_type": document_type,

        "chunks_stored": result["chunks_stored"],

        "category": result["category"],

        "summary": result["summary"],

        "company": result["company"],
    }


# ============================================================
# 6. Upload document from URL
# ============================================================

@app.post("/documents/upload-url")
def upload_document_from_url(
    request: URLRequest
):

    url = request.url.strip()

    # --------------------------------------------------------
    # Download URL
    # --------------------------------------------------------

    try:

        response = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/151.0.0.0 Safari/537.36"
                )
            },
        )

        response.raise_for_status()

    except requests.RequestException as exc:

        logger.error(
            f"URL download failed: {url} — {exc}"
        )

        raise HTTPException(
            status_code=400,
            detail=f"Could not download URL: {exc}"
        )


    # --------------------------------------------------------
    # Detect content type
    # --------------------------------------------------------

    content_type = response.headers.get(
        "content-type",
        ""
    ).lower()


    # ========================================================
    # CASE 1: PDF URL
    # ========================================================

    if (
        "application/pdf" in content_type
        or url.lower().split("?")[0].endswith(".pdf")
    ):

        filename = (
            url.rstrip("/")
            .split("/")[-1]
            .split("?")[0]
        )

        if not filename.lower().endswith(".pdf"):

            filename = "downloaded_document.pdf"

        source_hash = hashlib.sha256(
            response.content
        ).hexdigest()

        # Check whether this exact file is already indexed
        duplicate_name = find_duplicate(
            source_hash=source_hash
        )

        if duplicate_name:

            logger.info(
                f"URL upload duplicate skipped: {url} "
                f"(already indexed as '{duplicate_name}')"
            )

            return {

                "filename": filename,

                "source_url": url,

                "document_type": "pdf",

                "duplicate": True,

                "message": (
                    f"Already indexed as '{duplicate_name}', "
                    "skipped."
                ),

                "chunks_stored": 0,
            }

        file_path = UPLOAD_DIR / filename

        # Save PDF
        with file_path.open("wb") as buffer:

            buffer.write(
                response.content
            )

        # Ingest PDF
        result = ingest_document(
            str(file_path),
            source_hash=source_hash,
            company=request.company,
        )

        logger.info(
            f"URL upload indexed: {url} "
            f"({result['chunks_stored']} chunks, "
            f"category={result['category']}, "
            f"company={request.company})"
        )

        return {

            "filename": filename,

            "source_url": url,

            "document_type": "pdf",

            "duplicate": False,

            "message": (
                "PDF downloaded and "
                "indexed successfully"
            ),

            "chunks_stored": result["chunks_stored"],

            "category": result["category"],

            "summary": result["summary"],

            "company": result["company"],
        }


    # ========================================================
    # CASE 2: HTML webpage
    # ========================================================

    if "text/html" in content_type:

        # Check whether this exact URL is already indexed
        duplicate_name = find_duplicate(
            url=url
        )

        if duplicate_name:

            logger.info(
                f"URL upload duplicate skipped: {url} "
                f"(already indexed as '{duplicate_name}')"
            )

            return {

                "filename": duplicate_name,

                "source_url": url,

                "document_type": "webpage",

                "duplicate": True,

                "message": (
                    f"Already indexed as '{duplicate_name}', "
                    "skipped."
                ),

                "chunks_stored": 0,
            }

        try:

            # Extract webpage
            content = extract_webpage(
                url
            )

            # Chunk webpage
            chunks = chunk_document(
                content
            )

            # Classify + summarize (webpages don't go through
            # process_document(), so this doesn't happen for
            # them automatically the way it does for PDFs)
            chunks = classify_and_summarize(
                chunks
            )

            for chunk in chunks:
                chunk["company"] = request.company

            # Store in ChromaDB
            # (embeddings are computed internally by the vectorstore)
            stored_chunks = store_chunks(
                chunks
            )

        except Exception as exc:

            logger.error(
                f"Webpage processing failed: {url} — {exc}"
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    f"Could not process webpage: "
                    f"{exc}"
                )
            )

        logger.info(
            f"URL upload indexed: {url} "
            f"({stored_chunks} chunks, "
            f"category={chunks[0]['category'] if chunks else 'Other'}, "
            f"company={request.company})"
        )

        return {

            "filename": content[0]["document_name"],

            "source_url": url,

            "document_type": "webpage",

            "duplicate": False,

            "message": (
                "Webpage downloaded and "
                "indexed successfully"
            ),

            "chunks_stored": stored_chunks,

            "category": (
                chunks[0]["category"] if chunks else "Other"
            ),

            "summary": (
                chunks[0]["summary"] if chunks else ""
            ),

            "company": request.company,
        }


    # ========================================================
    # CASE 3: Unsupported content
    # ========================================================

    raise HTTPException(
        status_code=400,
        detail=(
            "Unsupported URL content type: "
            f"{content_type}"
        )
    )


# ============================================================
# 7. List documents in ChromaDB
# ============================================================

@app.get("/documents")
def get_documents():

    documents = list_documents()

    total_chunks = sum(
        doc["chunks"] for doc in documents
    )

    return {

        "documents": documents,

        "total_chunks": total_chunks,
    }


# ============================================================
# 8. Delete a document from ChromaDB
# ============================================================

@app.delete("/documents")
def remove_document(
    request: DocumentDeleteRequest
):

    document_name = request.document_name

    deleted_count = delete_document(document_name)

    if deleted_count == 0:

        logger.warning(
            f"Delete failed, not found: {document_name}"
        )

        raise HTTPException(
            status_code=404,
            detail=(
                "No chunks found for document "
                f"'{document_name}'"
            ),
        )

    logger.info(
        f"Deleted: {document_name} ({deleted_count} chunks)"
    )

    return {

        "document_name": document_name,

        "message": "Document deleted successfully",

        "chunks_deleted": deleted_count,
    }


# ============================================================
# 9. Ask question
# ============================================================

def stream_ndjson(event: dict) -> str:
    """
    One line of newline-delimited JSON. NDJSON rather than full
    Server-Sent Events — no extra protocol ceremony needed since
    both ends of this stream are ours (FastAPI -> Streamlit).
    """

    return json.dumps(event) + "\n"


def run_ask_pipeline(request: QuestionRequest):
    """
    Generator version of the ask pipeline: yields one status
    line per real stage as it actually completes, then a final
    result line. Every stage here is genuine — if the guardrail
    skips the LLM, fewer lines are yielded, not a fixed fake
    sequence.
    """

    start_time = time.time()

    # ----------------------------------------------------------
    # Step 1: Retrieve relevant chunks
    # ----------------------------------------------------------

    yield stream_ndjson({
        "type": "status",
        "message": "Retrieving relevant documents...",
    })

    retrieved_chunks = retrieve_documents(
        request.question,
        top_k=5,
    )


    # ----------------------------------------------------------
    # Step 2: Guardrail — check confidence BEFORE calling the
    # LLM. "Critical" band has, in every case tested, meant the
    # question is genuinely unanswerable from this corpus — so
    # skip the paid, slower LLM call entirely rather than paying
    # for it just to get the same refusal back. "Warn" band is
    # deliberately NOT short-circuited here: it's genuinely
    # ambiguous (a borderline retrieval score can still lead to
    # either a real answer or a refusal), so that judgment call
    # is left to the LLM rather than guessed at from confidence
    # alone.
    # ----------------------------------------------------------

    confidence = compute_confidence(retrieved_chunks)

    yield stream_ndjson({
        "type": "status",
        "message": (
            f"Found {len(retrieved_chunks)} chunks — "
            f"confidence {confidence['percent']}% "
            f"({confidence['band']})"
        ),
    })

    if confidence["band"] == "critical":

        elapsed = time.time() - start_time

        logger.info(
            f"Ask (guardrail skipped LLM): "
            f"{request.question!r} | "
            f"confidence={confidence['percent']}% | "
            f"{elapsed:.2f}s"
        )

        yield stream_ndjson({
            "type": "status",
            "message": (
                "Confidence too low — skipping answer "
                f"generation ({elapsed:.2f}s)"
            ),
        })

        yield stream_ndjson({
            "type": "result",
            "data": {
                "question": request.question,
                "answer": NO_ANSWER_MESSAGE,
                "sources": [],
                "confidence": confidence,
            },
        })

        return


    # ----------------------------------------------------------
    # Step 3: Send chunks to LLM
    # ----------------------------------------------------------

    yield stream_ndjson({
        "type": "status",
        "message": "Generating answer...",
    })

    answer = call_llm(
        request.question,
        retrieved_chunks,
        chat_history=[
            turn.model_dump()
            for turn in request.chat_history
        ],
    )


    # ----------------------------------------------------------
    # Step 4: Prepare sources
    # ----------------------------------------------------------

    sources = []

    for chunk in retrieved_chunks:

        source = {

            "document_name": (
                chunk["document_name"]
            ),

            "page_number": (
                chunk["page_number"]
            ),

            "content_type": (
                chunk["content_type"]
            ),

            "chunk_id": (
                chunk["chunk_id"]
            ),

            "text": (
                chunk["text"]
            ),
        }

        if "url" in chunk:
            source["url"] = chunk["url"]

        sources.append(source)


    # ----------------------------------------------------------
    # Step 5: The LLM can still refuse even when retrieval
    # wasn't confidently bad enough to skip calling it (e.g. a
    # borderline "warn"-band retrieval). When that happens,
    # correct the confidence and sources after the fact — they
    # were computed from retrieval alone and don't know the LLM
    # ended up finding nothing usable in them.
    # ----------------------------------------------------------

    elapsed = time.time() - start_time

    if answer.strip() == NO_ANSWER_MESSAGE:

        confidence = {
            "percent": 0,
            "band": "critical",
        }

        sources = []

        yield stream_ndjson({
            "type": "status",
            "message": (
                "Model found no usable answer in the "
                f"retrieved content ({elapsed:.2f}s)"
            ),
        })

    else:

        yield stream_ndjson({
            "type": "status",
            "message": (
                f"Answer generated ({len(sources)} sources) "
                f"in {elapsed:.2f}s"
            ),
        })


    # ----------------------------------------------------------
    # Step 6: Yield the final result
    # ----------------------------------------------------------

    logger.info(
        f"Ask (LLM called): "
        f"{request.question!r} | "
        f"confidence={confidence['percent']}% | "
        f"sources={len(sources)} | "
        f"{elapsed:.2f}s"
    )

    yield stream_ndjson({
        "type": "result",
        "data": {
            "question": request.question,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
        },
    })


@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    return StreamingResponse(
        run_ask_pipeline(request),
        media_type="application/x-ndjson",
    )


# ============================================================
# 10. Compare two companies
# ============================================================

def run_compare_pipeline(request: CompareRequest):
    """
    Same streaming shape as run_ask_pipeline, but retrieves each
    company separately (so one company's documents can't crowd
    out the other's) and reports confidence per side rather than
    blended into one number.
    """

    start_time = time.time()

    yield stream_ndjson({
        "type": "status",
        "message": f"Retrieving {request.company_a}'s documents...",
    })

    chunks_a = retrieve_documents(
        request.question,
        top_k=5,
        company=request.company_a,
    )

    confidence_a = compute_confidence(chunks_a)

    yield stream_ndjson({
        "type": "status",
        "message": f"Retrieving {request.company_b}'s documents...",
    })

    chunks_b = retrieve_documents(
        request.question,
        top_k=5,
        company=request.company_b,
    )

    confidence_b = compute_confidence(chunks_b)

    yield stream_ndjson({
        "type": "status",
        "message": (
            f"{request.company_a}: {confidence_a['percent']}% "
            f"confidence ({len(chunks_a)} chunks) — "
            f"{request.company_b}: {confidence_b['percent']}% "
            f"confidence ({len(chunks_b)} chunks)"
        ),
    })

    # Guardrail: only skip the LLM if NEITHER side has anything
    # worth comparing. One side being critical while the other
    # is fine is a real, useful case (verified during testing) —
    # not something to guard against.
    if (
        confidence_a["band"] == "critical"
        and confidence_b["band"] == "critical"
    ):

        elapsed = time.time() - start_time

        logger.info(
            f"Compare (guardrail skipped LLM): "
            f"{request.question!r} | both sides critical | "
            f"{elapsed:.2f}s"
        )

        yield stream_ndjson({
            "type": "status",
            "message": (
                "Neither company has relevant content — "
                f"skipping comparison ({elapsed:.2f}s)"
            ),
        })

        yield stream_ndjson({
            "type": "result",
            "data": {
                "question": request.question,
                "company_a": request.company_a,
                "company_b": request.company_b,
                "answer": NO_ANSWER_MESSAGE,
                "sources": [],
                "confidence_a": confidence_a,
                "confidence_b": confidence_b,
            },
        })

        return

    yield stream_ndjson({
        "type": "status",
        "message": "Generating comparison...",
    })

    answer = compare_answer(
        request.question,
        chunks_a,
        request.company_a,
        chunks_b,
        request.company_b,
    )

    # Combined, continuously-numbered source list — matches the
    # numbering compare_answer() used when building its prompt.
    sources = []

    for chunk, company in (
        [(c, request.company_a) for c in chunks_a]
        + [(c, request.company_b) for c in chunks_b]
    ):

        source = {
            "document_name": chunk["document_name"],
            "page_number": chunk["page_number"],
            "content_type": chunk["content_type"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "company": company,
        }

        if "url" in chunk:
            source["url"] = chunk["url"]

        sources.append(source)

    elapsed = time.time() - start_time

    logger.info(
        f"Compare (LLM called): {request.question!r} | "
        f"{request.company_a}={confidence_a['percent']}% | "
        f"{request.company_b}={confidence_b['percent']}% | "
        f"{elapsed:.2f}s"
    )

    yield stream_ndjson({
        "type": "status",
        "message": (
            f"Comparison generated ({len(sources)} sources) "
            f"in {elapsed:.2f}s"
        ),
    })

    yield stream_ndjson({
        "type": "result",
        "data": {
            "question": request.question,
            "company_a": request.company_a,
            "company_b": request.company_b,
            "answer": answer,
            "sources": sources,
            "confidence_a": confidence_a,
            "confidence_b": confidence_b,
        },
    })


@app.post("/compare")
def compare_documents(
    request: CompareRequest
):

    return StreamingResponse(
        run_compare_pipeline(request),
        media_type="application/x-ndjson",
    )


# ============================================================
# 11. Compare two specific documents ("what changed")
# ============================================================

def run_compare_documents_pipeline(
    request: CompareDocumentsRequest
):
    """
    Unlike run_compare_pipeline (topic search, per side), this
    fetches whole documents directly by name — there's no
    natural search query for "what's different between these
    two." No confidence meter here either: confidence was a
    retrieval-relevance concept, and this isn't a relevance
    search, so one would be meaningless. The trust signal
    instead is an honest truncation notice when a document is
    too large to send in full.
    """

    start_time = time.time()

    yield stream_ndjson({
        "type": "status",
        "message": (
            f"Fetching all chunks for {request.document_a}..."
        ),
    })

    result_a = get_document_chunks(request.document_a)

    yield stream_ndjson({
        "type": "status",
        "message": (
            f"Fetching all chunks for {request.document_b}..."
        ),
    })

    result_b = get_document_chunks(request.document_b)

    if result_a["total_chunks"] == 0 or result_b["total_chunks"] == 0:

        missing = (
            request.document_a
            if result_a["total_chunks"] == 0
            else request.document_b
        )

        elapsed = time.time() - start_time

        logger.warning(
            f"Compare-documents skipped: '{missing}' has no "
            f"indexed content | {elapsed:.2f}s"
        )

        yield stream_ndjson({
            "type": "status",
            "message": f"'{missing}' has no indexed content.",
        })

        yield stream_ndjson({
            "type": "result",
            "data": {
                "document_a": request.document_a,
                "document_b": request.document_b,
                "answer": (
                    f"'{missing}' has no indexed content to "
                    "compare."
                ),
                "sources": [],
                "truncated_a": False,
                "truncated_b": False,
                "total_chunks_a": result_a["total_chunks"],
                "total_chunks_b": result_b["total_chunks"],
            },
        })

        return

    truncation_notes = []

    if result_a["truncated"]:

        truncation_notes.append(
            f"{request.document_a}: using first "
            f"{len(result_a['chunks'])} of "
            f"{result_a['total_chunks']} chunks"
        )

    if result_b["truncated"]:

        truncation_notes.append(
            f"{request.document_b}: using first "
            f"{len(result_b['chunks'])} of "
            f"{result_b['total_chunks']} chunks"
        )

    if truncation_notes:

        yield stream_ndjson({
            "type": "status",
            "message": "Note: " + "; ".join(truncation_notes),
        })

    yield stream_ndjson({
        "type": "status",
        "message": "Generating comparison...",
    })

    focus = request.focus.strip() or None

    answer = compare_document_versions(
        request.document_a,
        result_a["chunks"],
        request.document_b,
        result_b["chunks"],
        focus=focus,
    )

    sources = []

    for chunk in (
        result_a["chunks"] + result_b["chunks"]
    ):

        source = {
            "document_name": chunk["document_name"],
            "page_number": chunk["page_number"],
            "content_type": chunk["content_type"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
        }

        if "url" in chunk:
            source["url"] = chunk["url"]

        sources.append(source)

    elapsed = time.time() - start_time

    logger.info(
        f"Compare-documents (LLM called): "
        f"{request.document_a!r} vs {request.document_b!r} | "
        f"{elapsed:.2f}s"
    )

    yield stream_ndjson({
        "type": "status",
        "message": (
            f"Comparison generated ({len(sources)} sources) "
            f"in {elapsed:.2f}s"
        ),
    })

    yield stream_ndjson({
        "type": "result",
        "data": {
            "document_a": request.document_a,
            "document_b": request.document_b,
            "answer": answer,
            "sources": sources,
            "truncated_a": result_a["truncated"],
            "truncated_b": result_b["truncated"],
            "total_chunks_a": result_a["total_chunks"],
            "total_chunks_b": result_b["total_chunks"],
        },
    })


@app.post("/compare-documents")
def compare_specific_documents(
    request: CompareDocumentsRequest
):

    return StreamingResponse(
        run_compare_documents_pipeline(request),
        media_type="application/x-ndjson",
    )