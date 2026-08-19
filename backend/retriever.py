import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from chroma_db import vectorstore
from embedding import embeddings


# 1. Retrieve relevant chunks, keeping each one's raw distance
#    score so answer confidence can be computed from it later.
#    (similarity_search_with_score returns (Document, distance)
#    pairs — lower distance means a closer/better match.)

def retrieve_documents(
    query: str,
    top_k: int = 5,
    company: str | None = None,
) -> list[dict]:
    """
    company, when given, restricts the search to only that
    company's chunks (used by the Compare feature, which needs
    to search each side separately rather than one pooled top-k
    — otherwise one company's documents could crowd out the
    other's entirely).

    This does NOT use Chroma's native `where` filter combined
    with vector search — that combination throws an internal
    error in this chromadb version regardless of which field is
    filtered on (verified directly, not specific to "company").
    Instead, this searches the whole collection unfiltered and
    filters/truncates to top_k in Python, which is only a little
    more work at this corpus size (low hundreds of chunks).
    """

    if company:

        total_chunks = vectorstore._collection.count()

        all_results = vectorstore.similarity_search_with_score(
            query,
            k=total_chunks,
        )

        results = [
            (document, distance)
            for document, distance in all_results
            if document.metadata.get("company") == company
        ][:top_k]

    else:

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


# 1b. Hybrid retrieval: combine vector search with BM25 keyword
#     search via Reciprocal Rank Fusion (RRF), so a chunk that
#     matches on exact keywords but only scores mediocre on
#     embedding similarity can still surface. This is what fixed
#     the reproduced failure case where a webpage chunk containing
#     the literal words "Atlassian" and "software" wasn't being
#     retrieved by vector search alone.
#
#     RRF combines *rank position* from each method rather than
#     raw scores, since embedding distances and BM25 scores are on
#     incompatible scales and can't be blended directly.
#
#     Confidence is still derived from each chunk's real vector
#     distance (looked up from the same full vector search used
#     for fusion), regardless of whether vector search or BM25 is
#     what actually surfaced it into the final top_k — this reuses
#     the existing, already-calibrated confidence system instead of
#     needing a second calibration for RRF's own score scale.

RRF_K = 60
FUSION_POOL_SIZE = 20
RERANK_POOL_SIZE = 20


def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    company: str | None = None,
) -> list[dict]:

    total_chunks = vectorstore._collection.count()

    all_vector_results = vectorstore.similarity_search_with_score(
        query,
        k=total_chunks,
    )

    if company:

        all_vector_results = [
            (document, distance)
            for document, distance in all_vector_results
            if document.metadata.get("company") == company
        ]

    documents_by_id = {
        document.id: document
        for document, _distance in all_vector_results
    }

    distance_by_id = {
        document.id: distance
        for document, distance in all_vector_results
    }

    vector_rank_by_id = {
        document.id: rank
        for rank, (document, _distance) in enumerate(
            all_vector_results[:FUSION_POOL_SIZE],
            start=1,
        )
    }

    # BM25 keyword search over the same (possibly company-scoped)
    # corpus. Rebuilt fresh on every call — the corpus is small
    # enough (low hundreds of chunks) that this is cheap, and it
    # avoids having to keep a cached index in sync with ingestion.
    corpus_ids = list(documents_by_id.keys())

    tokenized_corpus = [
        documents_by_id[chunk_id].page_content.lower().split()
        for chunk_id in corpus_ids
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    bm25_scores = bm25.get_scores(
        query.lower().split()
    )

    bm25_ranked_ids = [
        corpus_ids[i]
        for i in bm25_scores.argsort()[::-1]
    ][:FUSION_POOL_SIZE]

    bm25_rank_by_id = {
        chunk_id: rank
        for rank, chunk_id in enumerate(bm25_ranked_ids, start=1)
    }

    candidate_ids = set(vector_rank_by_id) | set(bm25_rank_by_id)

    def rrf_score(chunk_id: str) -> float:

        score = 0.0

        if chunk_id in vector_rank_by_id:
            score += 1 / (RRF_K + vector_rank_by_id[chunk_id])

        if chunk_id in bm25_rank_by_id:
            score += 1 / (RRF_K + bm25_rank_by_id[chunk_id])

        return score

    fused_ids = sorted(
        candidate_ids,
        key=rrf_score,
        reverse=True,
    )[:RERANK_POOL_SIZE]

    candidate_chunks = []

    for chunk_id in fused_ids:

        document = documents_by_id[chunk_id]
        metadata = document.metadata

        chunk = {
            "chunk_id": chunk_id,
            "text": document.page_content,
            "document_name": metadata["document_name"],
            "page_number": metadata["page_number"],
            "content_type": metadata["content_type"],
            "score": distance_by_id[chunk_id],
        }

        if "url" in metadata:
            chunk["url"] = metadata["url"]

        candidate_chunks.append(chunk)

    return rerank(query, candidate_chunks, top_k)


# 1c. Rerank the fused candidate pool with a cross-encoder: a
#     small model that reads the query and a chunk's text
#     TOGETHER and scores how well that specific chunk answers
#     that specific question — a more precise (but slower) signal
#     than rank position alone, which only ever compares each
#     candidate to the query in isolation. Only run over the
#     already-narrowed pool from hybrid_retrieve (not the whole
#     collection), since it's too slow to run on every chunk in
#     the corpus for every query. The final top_k pick is handed
#     off to mmr_select() below, rather than just taking the
#     highest-scored top_k directly.

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int,
) -> list[dict]:

    if not candidates:
        return candidates

    pairs = [
        (query, candidate["text"])
        for candidate in candidates
    ]

    relevance_scores = reranker.predict(pairs)

    return mmr_select(candidates, relevance_scores, top_k)


# 1d. Maximal Marginal Relevance (MMR): picks the final top_k from
#     the reranked pool by balancing relevance against redundancy,
#     so near-duplicate chunks (the same clause repeated across
#     multiple documents) don't crowd out genuinely different
#     supporting information. Picks the single most relevant chunk
#     first; each subsequent pick trades off relevance against how
#     similar it is to chunks already chosen, measured with the
#     same embedding model used everywhere else in retrieval
#     (already unit-normalized, so a plain dot product IS cosine
#     similarity — no extra normalization needed).

MMR_LAMBDA = 0.7


def mmr_select(
    candidates: list[dict],
    relevance_scores,
    top_k: int,
    lambda_param: float = MMR_LAMBDA,
) -> list[dict]:

    if not candidates:
        return candidates

    candidate_vectors = np.array(
        embeddings.embed_documents(
            [candidate["text"] for candidate in candidates]
        )
    )

    scores = np.array(relevance_scores, dtype=float)
    score_range = scores.max() - scores.min()

    # Cross-encoder scores aren't on a fixed scale, so they're
    # min-max normalized to 0-1 within this candidate pool before
    # being blended with cosine similarity (also 0-1-ish) — same
    # reasoning as using RANK instead of raw score for RRF fusion
    # earlier: two differently-scaled signals can't be blended
    # directly.
    normalized_relevance = (
        (scores - scores.min()) / score_range
        if score_range > 0
        else np.ones_like(scores)
    )

    remaining_indices = list(range(len(candidates)))
    selected_indices = []

    while remaining_indices and len(selected_indices) < top_k:

        if not selected_indices:

            best_index = max(
                remaining_indices,
                key=lambda i: normalized_relevance[i],
            )

        else:

            def mmr_score(i):

                similarity_to_selected = max(
                    candidate_vectors[i] @ candidate_vectors[j]
                    for j in selected_indices
                )

                return (
                    lambda_param * normalized_relevance[i]
                    - (1 - lambda_param) * similarity_to_selected
                )

            best_index = max(remaining_indices, key=mmr_score)

        selected_indices.append(best_index)
        remaining_indices.remove(best_index)

    return [candidates[i] for i in selected_indices]


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