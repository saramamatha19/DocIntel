import chromadb

from embedding import model


# 1. Connect to existing ChromaDB

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "docintel_documents"

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# 2. Retrieve relevant chunks

def retrieve_documents(
    query: str,
    top_k: int = 5,
) -> list[dict]:

    # Convert the user's question into an embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    retrieved_chunks = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    for document, metadata, distance, chunk_id in zip(
        documents,
        metadatas,
        distances,
        ids,
    ):

        chunk = {
            "chunk_id": chunk_id,
            "text": document,
            "document_name": metadata["document_name"],
            "page_number": metadata["page_number"],
            "content_type": metadata["content_type"],
            "distance": distance,
        }

        # Preserve webpage URL if available
        if "url" in metadata:
            chunk["url"] = metadata["url"]

        retrieved_chunks.append(chunk)

    return retrieved_chunks


# 3. Test retrieval

if __name__ == "__main__":

    question = (
        "What are Atlassian's responsibilities "
        "in the cloud security shared responsibility model?"
    )

    results = retrieve_documents(
        question,
        top_k=5,
    )

    print("\n==============================")
    print("RETRIEVAL RESULTS")
    print("==============================")

    for index, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n--- Result {index} ---"
        )

        print(
            f"Chunk ID: {result['chunk_id']}"
        )

        print(
            f"Document: {result['document_name']}"
        )

        print(
            f"Page: {result['page_number']}"
        )

        print(
            f"Content type: {result['content_type']}"
        )

        if "url" in result:
            print(
                f"URL: {result['url']}"
            )

        print(
            f"Distance: {result['distance']}"
        )

        print(
            f"\n{result['text']}"
        )