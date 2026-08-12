from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from retriever import retrieve_documents

# 1. Load environment variables
load_dotenv()

# 2. Create the chat model (via LangChain)
llm = ChatOpenAI(model="gpt-4o-mini")

# 3. Prompt template — same rules/format as before, expressed
#    as a LangChain ChatPromptTemplate instead of a raw f-string
PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
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
- Check the "Content type" line of the chunk you are citing, and
  follow the matching rule below:

  - If Content type is "webpage":
    You MUST cite the exact value that appears after "URL:" in that
    chunk. Do NOT cite the "Document" field for webpage content,
    even though a document name is also present.
    **Source:** https://example.com/page

  - If Content type is "text", "table", or "image_ocr" (i.e. any
    PDF-derived content, there is no "URL:" line):
    Cite the document name and page number.
    **Source:** Document_Name.pdf, Page X.

- Do not cite a URL just because the URL appears inside a PDF.
- Do not cite a document or page that does not support the statement.
- Do not create or guess page numbers.

CONVERSATION HISTORY:

- Use the conversation history ONLY to understand what the user is
  referring to (e.g. pronouns like "it", or follow-up phrases like
  "what about page 3?").
- Do NOT treat the conversation history as a source of facts. Facts
  must come only from the document context below.

Conversation so far (most recent last):
{chat_history}

FORMAT:

### Topic 1

Answer based only on the retrieved context.

**Source:** Document_Name.pdf, Page X.

### Topic 2 (example when the source is a webpage)

Answer based only on the retrieved context.

**Source:** https://example.com/page

If multiple pages from the same document support a topic,
you may cite them together:

**Source:** Document_Name.pdf, Pages X-Y.


User question:
{question}

Document context:
{context}
"""
)

# 4. Compose the LCEL chain: prompt -> chat model -> plain string
chain = PROMPT_TEMPLATE | llm | StrOutputParser()


# 5. Format prior turns for the prompt
def format_chat_history(
    chat_history: list[dict] | None,
) -> str:

    if not chat_history:
        return "No previous conversation."

    lines = []

    for turn in chat_history:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")

    return "\n".join(lines)


# 6. Call LLM with retrieved documents
def call_llm(
    question: str,
    retrieved_chunks: list[dict],
    chat_history: list[dict] | None = None,
) -> str:
    """
    Generate an answer using only the retrieved
    document chunks. `chat_history` (previous
    question/answer turns) is used only to resolve
    follow-up references, not as a source of facts.
    """

    context_parts = []

    for chunk in retrieved_chunks:

        url_line = (
            f'URL: {chunk["url"]}\n'
            if "url" in chunk
            else ""
        )

        context_parts.append(
            f"""
Document: {chunk["document_name"]}
Page: {chunk["page_number"]}
Content type: {chunk["content_type"]}
Chunk ID: {chunk["chunk_id"]}
{url_line}
{chunk["text"]}
"""
        )

    context = "\n---\n".join(context_parts)

    return chain.invoke({
        "question": question,
        "context": context,
        "chat_history": format_chat_history(chat_history),
    })

# 7. Run complete RAG pipeline
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