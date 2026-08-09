from sentence_transformers import SentenceTransformer

from document_ingestion import extract_document
from chunking import chunk_document


MODEL_NAME = "all-MiniLM-L6-v2"


# Load the embedding model once
model = SentenceTransformer(MODEL_NAME)


def create_embeddings(chunks: list[dict]) -> list[dict]:
    """
    Convert each chunk's text into an embedding vector
    while preserving the chunk metadata.
    """

    texts = [chunk["text"] for chunk in chunks]

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
    )

    embedded_chunks = []

    for chunk, vector in zip(chunks, vectors):

        embedded_chunks.append({
            **chunk,
            "embedding": vector.tolist(),
        })

    return embedded_chunks


if __name__ == "__main__":

    pdf_file = "uploads/GEP-Jun-2026-Regional-Highlights-MNA.pdf"

    # 1. Extract
    content = extract_document(pdf_file)

    # 2. Chunk
    chunks = chunk_document(content)

    # 3. Create embeddings
    embedded_chunks = create_embeddings(chunks)

    # 4. Inspect the first few
    for item in embedded_chunks[:3]:

        print("\n--- Embedded Chunk ---")

        print("Chunk ID:", item["chunk_id"])

        print("Content type:", item["content_type"])

        print("Page:", item["page_number"])

        print("Text:")
        print(item["text"][:200])

        print("Vector length:", len(item["embedding"]))

        print("First 10 numbers:")
        print(item["embedding"][:10])