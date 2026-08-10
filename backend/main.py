from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from chroma_db import ingest_pdf
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
# 3. Request model for questions
# ============================================================

class QuestionRequest(BaseModel):

    question: str


# ============================================================
# 4. Home endpoint
# ============================================================

@app.get("/")
def home():

    return {
        "message": "DocIntel Backend is Running!!!"
    }


# ============================================================
# 5. Upload document
# ============================================================

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    # Create file path

    file_path = UPLOAD_DIR / file.filename

    # Save uploaded PDF

    with file_path.open("wb") as buffer:

        buffer.write(
            await file.read()
        )

    # Ingest document into ChromaDB

    stored_chunks = ingest_pdf(
        str(file_path)
    )

    return {

        "filename": file.filename,

        "message": (
            "Document uploaded and "
            "indexed successfully"
        ),

        "chunks_stored": stored_chunks,
    }


# ============================================================
# 6. Ask question
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
    # Step 2: Send retrieved chunks to LLM
    # --------------------------------------------------------

    answer = call_llm(
        request.question,
        retrieved_chunks,
    )

    # --------------------------------------------------------
    # Step 3: Prepare sources
    # --------------------------------------------------------

    sources = []

    for chunk in retrieved_chunks:

        sources.append({

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
        })

    # --------------------------------------------------------
    # Step 4: Return answer + citations
    # --------------------------------------------------------

    return {

        "question": request.question,

        "answer": answer,

        "sources": sources,
    }