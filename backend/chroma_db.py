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

def process_document(file_path: str, source_hash: str | None = None):

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

    # Tag every chunk with the source file's hash so duplicate
    # uploads (even under a different filename) can be detected.
    if source_hash:

        for chunk in chunks:
            chunk["source_hash"] = source_hash

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

        if "source_hash" in chunk:
            metadata["source_hash"] = chunk["source_hash"]

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

def ingest_document(file_path: str, source_hash: str | None = None):

    chunks = process_document(
        file_path,
        source_hash=source_hash,
    )

    stored_count = store_chunks(
        chunks
    )

    return stored_count


# 5b. Check whether a document is already indexed

def find_duplicate(
    source_hash: str | None = None,
    url: str | None = None,
) -> str | None:
    """
    Return the document_name of an already-indexed document that
    matches the given source_hash (file uploads, hashed on the raw
    bytes) or url (webpages, matched on the exact URL string).
    Returns None if nothing matches either.
    """

    if source_hash:

        matches = vectorstore._collection.get(
            where={"source_hash": source_hash},
            include=["metadatas"],
        )

        if matches["ids"]:
            return matches["metadatas"][0]["document_name"]

    if url:

        matches = vectorstore._collection.get(
            where={"url": url},
            include=["metadatas"],
        )

        if matches["ids"]:
            return matches["metadatas"][0]["document_name"]

    return None


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


# 7. List indexed documents

def list_documents():

    data = vectorstore._collection.get(
        include=["metadatas"]
    )

    counts = {}

    for metadata in data["metadatas"]:

        name = metadata.get("document_name")

        counts[name] = counts.get(name, 0) + 1

    return [
        {
            "document_name": name,
            "chunks": count,
        }
        for name, count in sorted(counts.items())
    ]


# 8. Delete one document (all its chunks, plus the uploaded
#    file if it was a local upload rather than a webpage)

def delete_document(document_name: str) -> int:

    matches = vectorstore._collection.get(
        where={"document_name": document_name},
        include=[],
    )

    deleted_count = len(matches["ids"])

    if deleted_count == 0:
        return 0

    vectorstore.delete(
        where={"document_name": document_name}
    )

    # Webpage entries use a URL-shaped document_name and have
    # no corresponding file on disk.
    if "/" not in document_name:

        upload_dir = Path("uploads").resolve()

        for file_path in upload_dir.rglob(document_name):

            if file_path.resolve().is_relative_to(upload_dir):
                file_path.unlink()

    return deleted_count


# 9. Run manually

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