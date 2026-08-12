import streamlit as st
import requests


# -----------------------------
# Configuration
# -----------------------------

API_URL = "http://127.0.0.1:8000"


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="DocIntel",
    page_icon="📚",
    layout="wide",
)


# -----------------------------
# Header
# -----------------------------

st.title("📚 DocIntel")

st.write(
    "AI-powered document intelligence and RAG assistant"
)


# -----------------------------
# Sidebar - Document Upload
# -----------------------------

st.sidebar.header("📄 Add Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"],
)


if st.sidebar.button("Upload & Index"):

    if uploaded_file is None:

        st.sidebar.warning(
            "Please select a PDF first."
        )

    else:

        with st.sidebar.spinner(
            "Uploading and indexing..."
        ):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf",
                )
            }

            response = requests.post(
                f"{API_URL}/documents/upload",
                files=files,
                timeout=120,
            )


        if response.ok:

            data = response.json()

            st.sidebar.success(
                "Document indexed successfully!"
            )

            st.sidebar.write(
                f"**File:** {data['filename']}"
            )

            st.sidebar.write(
                f"**Chunks:** {data['chunks_stored']}"
            )

        else:

            st.sidebar.error(
                f"Upload failed: {response.text}"
            )


# -----------------------------
# URL Upload
# -----------------------------

st.sidebar.header("🌐 Add PDF from URL")

pdf_url = st.sidebar.text_input(
    "PDF URL"
)


if st.sidebar.button("Download & Index"):

    if not pdf_url:

        st.sidebar.warning(
            "Please enter a PDF URL."
        )

    else:

        with st.sidebar.spinner(
            "Downloading and indexing..."
        ):

            response = requests.post(
                f"{API_URL}/documents/upload-url",
                json={
                    "url": pdf_url
                },
                timeout=120,
            )


        if response.ok:

            data = response.json()

            st.sidebar.success(
                "PDF indexed successfully!"
            )

            st.sidebar.write(
                f"**File:** {data['filename']}"
            )

            st.sidebar.write(
                f"**Chunks:** {data['chunks_stored']}"
            )

        else:

            st.sidebar.error(
                f"Download failed: {response.text}"
            )


# -----------------------------
# Main - Ask Question
# -----------------------------

st.header("💬 Ask your documents")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "question_input_id" not in st.session_state:
    st.session_state.question_input_id = 0


# -----------------------------
# Render previous turns (answers + sources)
# -----------------------------

for turn in st.session_state.chat_history:

    st.subheader("💡 Answer")

    st.write(
        turn["answer"]
    )

    st.subheader("📑 Sources")

    for index, source in enumerate(
        turn["sources"],
        start=1,
    ):

        with st.expander(
            f"Source {index}: "
            f"{source['document_name']} "
            f"(Page {source['page_number']})"
        ):

            st.write(
                f"**Document:** "
                f"{source['document_name']}"
            )

            st.write(
                f"**Page:** "
                f"{source['page_number']}"
            )

            st.write(
                f"**Content type:** "
                f"{source['content_type']}"
            )

            if "url" in source:

                st.write(
                    f"**URL:** "
                    f"{source['url']}"
                )

            st.write(
                f"**Chunk ID:** "
                f"{source['chunk_id']}"
            )

    st.divider()


question = st.text_area(
    "Ask a question or follow-up question about your documents",
    placeholder=(
        "Example: What are Atlassian's "
        "security responsibilities?"
    ),
    height=120,
    key=f"question_input_{st.session_state.question_input_id}",
)


if st.button("🔍 Ask"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            response = requests.post(
                f"{API_URL}/ask",
                json={
                    "question": question,
                    "chat_history": [
                        {
                            "question": turn["question"],
                            "answer": turn["answer"],
                        }
                        for turn in st.session_state.chat_history
                    ],
                },
                timeout=120,
            )


        if response.ok:

            data = response.json()

            st.session_state.chat_history.append({
                "question": question,
                "answer": data["answer"],
                "sources": data["sources"],
            })

            st.session_state.question_input_id += 1

            st.rerun()

        else:

            st.error(
                f"Request failed: {response.text}"
            )