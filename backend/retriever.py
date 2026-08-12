from chroma_db import vectorstore


# 1. Retrieve relevant chunks using LangChain's standard
#    Retriever interface

def retrieve_documents(
    query: str,
    top_k: int = 5,
) -> list[dict]:

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": top_k}
    )

    documents = retriever.invoke(query)

    retrieved_chunks = []

    for document in documents:

        metadata = document.metadata

        chunk = {
            "chunk_id": document.id,
            "text": document.page_content,
            "document_name": metadata["document_name"],
            "page_number": metadata["page_number"],
            "content_type": metadata["content_type"],
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
            f"\n{result['text']}"
        )