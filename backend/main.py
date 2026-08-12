from pathlib import Path

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from document_ingestion import extract_webpage
from chunking import chunk_document
from chroma_db import ingest_document, store_chunks

from retriever import retrieve_documents
from call_llm import call_llm


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


# ============================================================
# 4. Home endpoint
# ============================================================

@app.get("/")
def home():

    return {
        "message": "DocIntel Backend is Running!!!"
    }


# ============================================================
# 5. Upload PDF file
# ============================================================

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    # Check PDF
    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    # Create file path
    file_path = UPLOAD_DIR / file.filename

    # Save uploaded PDF
    with file_path.open("wb") as buffer:

        buffer.write(
            await file.read()
        )

    # Ingest PDF into ChromaDB
    stored_chunks = ingest_document(
        str(file_path)
    )

    return {

        "filename": file.filename,

        "message": (
            "Document uploaded and "
            "indexed successfully"
        ),

        "document_type": "pdf",

        "chunks_stored": stored_chunks,
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

        file_path = UPLOAD_DIR / filename

        # Save PDF
        with file_path.open("wb") as buffer:

            buffer.write(
                response.content
            )

        # Ingest PDF
        stored_chunks = ingest_document(
            str(file_path)
        )

        return {

            "filename": filename,

            "source_url": url,

            "document_type": "pdf",

            "message": (
                "PDF downloaded and "
                "indexed successfully"
            ),

            "chunks_stored": stored_chunks,
        }


    # ========================================================
    # CASE 2: HTML webpage
    # ========================================================

    if "text/html" in content_type:

        try:

            # Extract webpage
            content = extract_webpage(
                url
            )

            # Chunk webpage
            chunks = chunk_document(
                content
            )

            # Store in ChromaDB
            # (embeddings are computed internally by the vectorstore)
            stored_chunks = store_chunks(
                chunks
            )

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail=(
                    f"Could not process webpage: "
                    f"{exc}"
                )
            )

        return {

            "filename": content[0]["document_name"],

            "source_url": url,

            "document_type": "webpage",

            "message": (
                "Webpage downloaded and "
                "indexed successfully"
            ),

            "chunks_stored": stored_chunks,
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
# 7. Ask question
# ============================================================

@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    # --------------------------------------------------------
    # Step 1: Retrieve relevant chunks
    # --------------------------------------------------------

    retrieved_chunks = retrieve_documents(
        request.question,
        top_k=5,
    )


    # --------------------------------------------------------
    # Step 2: Send chunks to LLM
    # --------------------------------------------------------

    answer = call_llm(
        request.question,
        retrieved_chunks,
        chat_history=[
            turn.model_dump()
            for turn in request.chat_history
        ],
    )


    # --------------------------------------------------------
    # Step 3: Prepare sources
    # --------------------------------------------------------

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
        }

        if "url" in chunk:
            source["url"] = chunk["url"]

        sources.append(source)


    # --------------------------------------------------------
    # Step 4: Return answer + sources
    # --------------------------------------------------------

    return {

        "question": request.question,

        "answer": answer,

        "sources": sources,
    }