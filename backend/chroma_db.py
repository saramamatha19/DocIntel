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

# 3. Process one PDF
def process_pdf(pdf_file: str):

    print(f"\nProcessing: {pdf_file}")

    # Extract
    content = extract_document(pdf_file)

    print(f"Extracted items: {len(content)}")

    # Chunk
    chunks = chunk_document(content)

    print(f"Created chunks: {len(chunks)}")

    # Embed
    embedded_chunks = create_embeddings(chunks)

    print(f"Created embeddings: {len(embedded_chunks)}")

    return embedded_chunks

# 4. Store chunks in ChromaDB
def store_chunks(embedded_chunks):

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for chunk in embedded_chunks:

        ids.append(chunk["chunk_id"])

        documents.append(chunk["text"])

        metadatas.append({
            "document_name": chunk["document_name"],
            "page_number": chunk["page_number"],
            "content_type": chunk["content_type"],
        })

        embeddings.append(chunk["embedding"])

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Stored {len(ids)} chunks in ChromaDB")

# 5. Process all PDFs
def ingest_all_pdfs(folder: str):

    pdf_folder = Path(folder)

    pdf_files = list(pdf_folder.rglob("*.pdf"))

    print(f"\nFound {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:

        embedded_chunks = process_pdf(
            str(pdf_file)
        )

        store_chunks(
            embedded_chunks
        )

# 6. Run
if __name__ == "__main__":

    ingest_all_pdfs(
        "uploads/atlassian"
    )

    print("\n-----------------------------")
    print("ChromaDB ingestion complete")
    print("-----------------------------")

    print(
        "Total stored chunks:",
        collection.count()
    )