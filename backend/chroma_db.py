from pathlib import Path
import chromadb

from document_ingestion import extract_document
from chunking import chunk_document
from embedding import create_embeddings


# 1. ChromaDB configuration

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "docintel_documents"


# 2. Create persistent ChromaDB

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
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

    # Embed
    embedded_chunks = create_embeddings(
        chunks
    )

    print(
        f"Created embeddings: {len(embedded_chunks)}"
    )

    return embedded_chunks


# 4. Store chunks in ChromaDB

def store_chunks(embedded_chunks):

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for chunk in embedded_chunks:

        ids.append(
            chunk["chunk_id"]
        )

        documents.append(
            chunk["text"]
        )

        metadatas.append({
            "document_name": chunk["document_name"],
            "page_number": chunk["page_number"],
            "content_type": chunk["content_type"],
        })

        embeddings.append(
            chunk["embedding"]
        )

    if not ids:
        return 0

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(
        f"Stored {len(ids)} chunks in ChromaDB"
    )

    return len(ids)


# 5. Ingest ONE document

def ingest_document(file_path: str):

    embedded_chunks = process_document(
        file_path
    )

    stored_count = store_chunks(
        embedded_chunks
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
        collection.count()
    )