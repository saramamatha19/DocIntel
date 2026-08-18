from chroma_db import vectorstore


# 1. Retrieve relevant chunks, keeping each one's raw distance
#    score so answer confidence can be computed from it later.
#    (similarity_search_with_score returns (Document, distance)
#    pairs — lower distance means a closer/better match.)

def retrieve_documents(
    query: str,
    top_k: int = 5,
) -> list[dict]:

    results = vectorstore.similarity_search_with_score(
        query,
        k=top_k,
    )

    retrieved_chunks = []

    for document, distance in results:

        metadata = document.metadata

        chunk = {
            "chunk_id": document.id,
            "text": document.page_content,
            "document_name": metadata["document_name"],
            "page_number": metadata["page_number"],
            "content_type": metadata["content_type"],
            "score": distance,
        }

        # Preserve webpage URL if available
        if "url" in metadata:
            chunk["url"] = metadata["url"]

        retrieved_chunks.append(chunk)

    return retrieved_chunks


# 2. Convert a raw distance score into a 0-100 confidence value.
#
#    The floor/ceiling below are calibrated from real queries run
#    against this project's data, not a theoretical formula:
#    a clearly-relevant match scored ~0.37-0.73, while clearly
#    irrelevant queries scored ~1.2-1.8. This is a *relative*
#    retrieval-quality signal, not a calibrated probability that
#    the answer is correct.

CONFIDENCE_FLOOR = 0.3
CONFIDENCE_CEILING = 1.8


def score_to_confidence(distance: float) -> int:

    clamped = max(
        CONFIDENCE_FLOOR,
        min(distance, CONFIDENCE_CEILING),
    )

    ratio = (
        (CONFIDENCE_CEILING - clamped)
        / (CONFIDENCE_CEILING - CONFIDENCE_FLOOR)
    )

    return round(ratio * 100)


def confidence_band(percent: int) -> str:

    if percent >= 63:
        return "good"

    if percent >= 33:
        return "warn"

    return "critical"


# 3. Compute one overall confidence value for an answer, based
#    on the single best (lowest-distance) retrieved chunk.

def compute_confidence(retrieved_chunks: list[dict]) -> dict:

    if not retrieved_chunks:

        return {
            "percent": 0,
            "band": "critical",
        }

    best_score = min(
        chunk["score"] for chunk in retrieved_chunks
    )

    percent = score_to_confidence(best_score)

    return {
        "percent": percent,
        "band": confidence_band(percent),
    }


# 4. Test retrieval

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

        print(
            f"Score (raw distance): {result['score']:.4f}"
        )

        if "url" in result:
            print(
                f"URL: {result['url']}"
            )

        print(
            f"\n{result['text']}"
        )

    print("\n==============================")
    print("ANSWER CONFIDENCE")
    print("==============================")

    print(
        compute_confidence(results)
    )