from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 1. Load environment variables
load_dotenv()

# 2. Create the chat model (via LangChain)
llm = ChatOpenAI(model="gpt-4o-mini")

# Must stay in sync with the refusal instruction inside
# PROMPT_TEMPLATE below. main.py compares the LLM's raw
# output against this exact string to detect a "no answer"
# response and override its confidence display accordingly.
NO_ANSWER_MESSAGE = (
    "I could not find this information in the provided documents."
)

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

- Only split your answer into separate "### Topic" sections when
  the user's question itself asks about genuinely distinct things
  (e.g. "compare X and Y", or "what about A, and also B?").
- For a single, broad question (e.g. "tell me about X"), write one
  flowing answer instead — do NOT invent topic sections just
  because the supporting passages happen to come from different
  documents. Inline [N] citations already show which source backs
  each individual claim, so topic-splitting is not needed to avoid
  mixing sources together.
- Do NOT combine information from different documents into one
  unsupported claim.
- If a source's "Content type" is "table", never paste its raw
  pipe-delimited row text into your answer. Either state the
  specific fact needed as a normal sentence, or, if showing the
  whole table is genuinely useful, reformat it as a clean markdown
  table (using proper markdown table syntax) — not the raw
  extracted rows.

SOURCE CITATION RULES:

- Below, each retrieved passage is labeled "Source 1", "Source 2",
  and so on. Cite a source by writing its number in square brackets,
  e.g. [1], immediately after the specific clause or sentence it
  supports.
- Place the bracket marker inline, mid-sentence if needed. Do NOT
  collect citations together at the end of a paragraph.
- A single sentence may need more than one marker if two sources
  both support it, e.g. "...within 90 days [1][3]."
- Only cite a source number that actually supports the exact
  statement next to it. Do not cite a source that does not support
  the claim, and do not invent source numbers that were not given
  to you below.

CONVERSATION HISTORY:

- Use the conversation history ONLY to understand what the user is
  referring to (e.g. pronouns like "it", or follow-up phrases like
  "what about page 3?").
- Do NOT treat the conversation history as a source of facts. Facts
  must come only from the numbered sources below.

Conversation so far (most recent last):
{chat_history}

FORMAT EXAMPLES:

A single, broad question — write one flowing answer:

Atlassian provides cloud and software products for internal
business use [1]. It maintains a shared responsibility model in
which Atlassian secures the underlying infrastructure while
customers manage their own users and data [2][3].

A question that genuinely asks about multiple distinct things
(e.g. "compare X and Y", or "what about A, and also B?"):

### Topic 1

Atlassian retains customer data for 90 days after termination [1].

### Topic 2

Answer based only on the retrieved context [2][3].


User question:
{question}

Numbered sources:
{context}
"""
)

# 4. Compose the LCEL chain: prompt -> chat model -> plain string
chain = PROMPT_TEMPLATE | llm | StrOutputParser()


# 4b. Document classification — a separate, small prompt/chain
#     from the answer-generation one above. Adding a category
#     later is a one-line change to this list, nothing else.

DOCUMENT_CATEGORIES = [
    "HR",
    "Legal & Compliance",
    "Security & IT",
    "Engineering & Product",
    "Finance",
    "Sales & Marketing",
    "Operations & Facilities",
    "Executive & Strategy",
    "Other",
]

CLASSIFY_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
Classify the following document into exactly ONE of these
categories: {categories}.

Base your answer only on the excerpt below. Respond with only the
category name, exactly as written above — no punctuation, no
explanation.

Document excerpt:
{text_sample}
"""
)

classify_chain = (
    CLASSIFY_PROMPT_TEMPLATE | llm | StrOutputParser()
)


def classify_document(text_sample: str) -> str:
    """
    Classify a document into one of DOCUMENT_CATEGORIES using a
    small sample of its text (its first chunk is enough — a
    document's title/opening paragraph reveals its subject
    without needing to send the whole thing to the LLM).

    Falls back to "Other" if the model's raw output doesn't
    exactly match one of the known categories, so a bad/unknown
    value never ends up stored as a category.
    """

    raw_result = classify_chain.invoke({
        "categories": ", ".join(DOCUMENT_CATEGORIES),
        "text_sample": text_sample,
    })

    category = raw_result.strip()

    if category not in DOCUMENT_CATEGORIES:
        return "Other"

    return category


# 4c. Document summarization — its own small prompt/chain,
#     same shape as classification above.

SUMMARIZE_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
Summarize the following document in ONE sentence, focused on
what it specifically covers. Do not use generic filler like
"this document contains information about..." — state the
actual subject directly.

Document excerpt:
{text_sample}
"""
)

summarize_chain = (
    SUMMARIZE_PROMPT_TEMPLATE | llm | StrOutputParser()
)


def summarize_document(text_sample: str) -> str:
    """
    Generate a one-sentence summary of a document using a small
    sample of its text (its first chunk, same reasoning as
    classify_document — the opening content is representative
    enough without sending the whole document to the LLM).
    """

    summary = summarize_chain.invoke({
        "text_sample": text_sample,
    })

    return summary.strip()


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

    Sources are numbered in the same order as
    `retrieved_chunks`, so a [N] marker in the answer
    always corresponds to retrieved_chunks[N-1] — the
    same order main.py uses to build its `sources` list,
    so the frontend can map [N] straight to sources[N-1].
    """

    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):

        url_line = (
            f'URL: {chunk["url"]}\n'
            if "url" in chunk
            else ""
        )

        context_parts.append(
            f"""
Source {index}:
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

    from retriever import retrieve_documents

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