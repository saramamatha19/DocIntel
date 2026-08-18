from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_ingestion import extract_document
from pii_redaction import redact_pii


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

        text = redact_pii(item["text"].strip())

        if not text:
            continue

        content_type = item["content_type"]

        # ---------------------------------
        # TABLE
        # ---------------------------------

        if content_type == "table":

            chunk_id = (
                f"{item['document_name']}"
                f"_page{item['page_number']}"
                f"_table{item['table_number']}"
            )

            chunk = {
                "chunk_id": chunk_id,
                "text": text,
                "document_name": item["document_name"],
                "page_number": item["page_number"],
                "content_type": "table",
                "table_number": item["table_number"],
            }

            chunks.append(chunk)

            continue

        # ---------------------------------
        # TEXT / IMAGE OCR / WEBPAGE
        # ---------------------------------

        split_texts = text_splitter.split_text(text)

        for chunk_index, chunk_text in enumerate(
            split_texts,
            start=1,
        ):

            # IMAGE OCR needs image_number
            # in the chunk ID
            if content_type == "image_ocr":

                chunk_id = (
                    f"{item['document_name']}"
                    f"_page{item['page_number']}"
                    f"_image{item['image_number']}"
                    f"_ocr_{chunk_index}"
                )

            else:

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

            # Preserve image identity
            if content_type == "image_ocr":
                chunk["image_number"] = item["image_number"]

            # Preserve webpage URL
            if "url" in item:
                chunk["url"] = item["url"]

            chunks.append(chunk)

    return chunks


# ---------------------------------
# TEST
# ---------------------------------

if __name__ == "__main__":

    pdf_file = (
        "/Users/saramamatha/Projects_Sara/DocIntel/backend/uploads/it_support_policy_pii_test.txt"
    )

    # 1. Extract
    content = extract_document(pdf_file)

    # 2. Chunk
    chunks = chunk_document(content)

    # 3. Display
    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        print(
            f"\n--- Chunk {index} | "
            f"{chunk['content_type'].upper()} | "
            f"Page {chunk['page_number']} ---"
        )

        print(
            f"Document: {chunk['document_name']}"
        )

        print(
            f"Chunk ID: {chunk['chunk_id']}"
        )

        if "table_number" in chunk:
            print(
                f"Table: {chunk['table_number']}"
            )

        if "image_number" in chunk:
            print(
                f"Image: {chunk['image_number']}"
            )

        if "url" in chunk:
            print(
                f"URL: {chunk['url']}"
            )

        print(chunk["text"])