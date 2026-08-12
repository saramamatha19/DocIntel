from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from document_ingestion import extract_document
from chunking import chunk_document
from embedding import embeddings


# 1. ChromaDB configuration

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "docintel_documents"


# 2. Create persistent Chroma vector store (via LangChain)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)


# 3. Process ONE document

def process_document(file_path: str):

    print(f"\nProcessing: {file_path}")

    # Extract
    content = extract_document(file_path)

    print(
        f"Extracted items: {len(content)}"
    )

    # Chunk
    chunks = chunk_document(content)

    print(
        f"Created chunks: {len(chunks)}"
    )

    return chunks


# 4. Store chunks in the vector store
#    (embeddings are computed internally by the vectorstore)

def store_chunks(chunks):

    if not chunks:
        return 0

    ids = []
    documents = []

    for chunk in chunks:

        metadata = {
            "document_name": chunk["document_name"],
            "page_number": chunk["page_number"],
            "content_type": chunk["content_type"],
        }

        if "url" in chunk:
            metadata["url"] = chunk["url"]

        ids.append(chunk["chunk_id"])

        documents.append(
            Document(
                page_content=chunk["text"],
                metadata=metadata,
            )
        )

    vectorstore.add_documents(
        documents=documents,
        ids=ids,
    )

    print(
        f"Stored {len(ids)} chunks in ChromaDB"
    )

    return len(ids)


# 5. Ingest ONE document

def ingest_document(file_path: str):

    chunks = process_document(
        file_path
    )

    stored_count = store_chunks(
        chunks
    )

    return stored_count


# 6. Process ALL PDFs

def ingest_all_pdfs(folder: str):

    pdf_folder = Path(folder)

    pdf_files = list(
        pdf_folder.rglob("*.pdf")
    )

    print(
        f"\nFound {len(pdf_files)} PDF files"
    )

    total_stored = 0

    for pdf_file in pdf_files:

        stored_count = ingest_document(
            str(pdf_file)
        )

        total_stored += stored_count

    return total_stored


# 7. Run manually

if __name__ == "__main__":

    total_stored = ingest_all_pdfs(
        "uploads/atlassian"
    )

    print(
        "\n-----------------------------"
    )

    print(
        "ChromaDB ingestion complete"
    )

    print(
        "-----------------------------"
    )

    print(
        "Total stored chunks:",
        vectorstore._collection.count()
    )