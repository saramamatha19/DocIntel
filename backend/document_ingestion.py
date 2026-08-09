import pymupdf


def extract_text(file_path: str) -> str:
    document = pymupdf.open(file_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text

if __name__ == "__main__":
    file_path = "uploads/docintel_test_document.pdf"

    text = extract_text(file_path)

    print(text[:2000])