from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_ingestion import extract_document


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def chunk_document(content: list[dict]) -> list[dict]:
    """
    Split extracted document content into retrieval-friendly chunks
    while preserving document metadata.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = []

    for item in content:

        text = item["text"].strip()

        if not text:
            continue #if text is empty, skip to next item

        content_type = item["content_type"]

        # TABLE
        if content_type == "table":

            chunk_id = (
                f"{item['document_name']}"
                f"_page{item['page_number']}"
                f"_table{item['table_number']}"
            )

            chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "document_name": item["document_name"],
                "page_number": item["page_number"],
                "content_type": "table",
                "table_number": item["table_number"],
            })

            continue

        # TEXT / IMAGE OCR
        split_texts = text_splitter.split_text(text)

        for chunk_index, chunk_text in enumerate(
            split_texts,
            start=1,
        ):

            chunk_id = (
                f"{item['document_name']}"
                f"_page{item['page_number']}"
                f"_{content_type}"
                f"_{chunk_index}"
            )

            chunk = {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "document_name": item["document_name"],
                "page_number": item["page_number"],
                "content_type": content_type,
                "chunk_index": chunk_index,
            }

            # Preserve image identity for OCR content
            if content_type == "image_ocr":
                chunk["image_number"] = item["image_number"]

            chunks.append(chunk)

    return chunks


if __name__ == "__main__":

    pdf_file = "uploads/GEP-Jun-2026-Regional-Highlights-MNA.pdf"

    # 1. Extract
    content = extract_document(pdf_file)

    # 2. Chunk
    chunks = chunk_document(content)

    # 3. Display
    for index, chunk in enumerate(chunks, start=1):

        print(
            f"\n--- Chunk {index} | "
            f"{chunk['content_type'].upper()} | "
            f"Page {chunk['page_number']} ---"
        )

        print(f"Document: {chunk['document_name']}")
        print(f"Chunk ID: {chunk['chunk_id']}")

        if "table_number" in chunk:
            print(f"Table: {chunk['table_number']}")

        if "image_number" in chunk:
            print(f"Image: {chunk['image_number']}")

        print(chunk["text"])