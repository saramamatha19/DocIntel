from openai import OpenAI
from dotenv import load_dotenv
from retriever import retrieve_documents

# 1. Load environment variables
load_dotenv()

# 2. Create OpenAI client
client = OpenAI()

# 3. Call LLM with retrieved documents
def call_llm(
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    """
    Generate an answer using only the retrieved
    document chunks.
    """

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"""
Document: {chunk["document_name"]}
Page: {chunk["page_number"]}
Content type: {chunk["content_type"]}
Chunk ID: {chunk["chunk_id"]}

{chunk["text"]}
"""
        )

    context = "\n---\n".join(context_parts)

    prompt = f"""
You are an AI assistant that answers questions from company documents.

Answer the user's question using ONLY the provided document context.

Rules:

- Do not use outside knowledge.
- Do not invent or assume information.
- If the answer is not present in the provided context, say:
  "I could not find this information in the provided documents."

- If the question asks about multiple topics, answer each topic separately.
- If information comes from different documents, keep the topics separate.
- Do NOT combine information from different documents into one unsupported claim.

SOURCE CITATION RULES:

- Every factual section must have its own source.
- A single answer may contain multiple sources.
- For PDF content, cite:
  **Source:** Document_Name.pdf, Page X.
- For webpage content, cite:
  **Source:** URL.
- Do not cite a URL just because the URL appears inside a PDF.
- Use the document name and page number from the provided context.
- Do not cite a document or page that does not support the statement.
- Do not create or guess page numbers.

FORMAT:

### Topic 1

Answer based only on the retrieved context.

**Source:** Document_Name.pdf, Page X.

### Topic 2

Answer based only on the retrieved context.

**Source:** Document_Name.pdf, Page X.

If multiple pages from the same document support a topic,
you may cite them together:

**Source:** Document_Name.pdf, Pages X-Y.


User question:
{question}

Document context:
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content

# 4. Run complete RAG pipeline
if __name__ == "__main__":

    question = input("\nAsk a question: ")

    # Retrieve relevant chunks from ChromaDB
    retrieved_chunks = retrieve_documents(
        question,
        top_k=5,
    )

    print(
        f"\nRetrieved {len(retrieved_chunks)} relevant chunks."
    )

    # Send retrieved chunks to LLM
    answer = call_llm(
        question,
        retrieved_chunks,
    )

    print("\n==============================")
    print("ANSWER")
    print("==============================")

    print(answer)