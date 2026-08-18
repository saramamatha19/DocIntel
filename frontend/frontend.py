import streamlit as st
import requests

# Configuration
API_URL = "http://127.0.0.1:8000"

# Page configuration
st.set_page_config(
    page_title="DocIntel",
    page_icon="📚",
    layout="wide",
)

# Header
st.title("📚 DocIntel")

st.write(
    "AI-powered document intelligence and RAG assistant"
)


# Sidebar - Document Upload

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

            if data.get("duplicate"):

                st.sidebar.warning(
                    data["message"]
                )

            else:

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


# URL Upload

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

            if data.get("duplicate"):

                st.sidebar.warning(
                    data["message"]
                )

            else:

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


# ============================================================
# Main area - Chat / Documents tabs
# ============================================================

tab_chat, tab_documents = st.tabs([
    "💬 Chat",
    "📁 Documents",
])


# ------------------------------------------------------------
# Chat tab
# ------------------------------------------------------------

with tab_chat:

    st.header("💬 Ask your documents")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


    # Render previous turns (answers + sources)

    for turn in st.session_state.chat_history:

        with st.chat_message("user"):

            st.write(
                turn["question"]
            )

        with st.chat_message("assistant"):

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


    question = st.chat_input(
        "Enter your question here..."
    )


    if question:

        with st.chat_message("user"):

            st.write(
                question
            )

        with st.chat_message("assistant"):

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

                st.rerun()

            else:

                st.error(
                    f"Request failed: {response.text}"
                )


# ------------------------------------------------------------
# Documents tab
# ------------------------------------------------------------

with tab_documents:

    st.header("📁 Indexed documents")

    documents_response = requests.get(
        f"{API_URL}/documents",
        timeout=30,
    )

    if documents_response.ok:

        documents_data = documents_response.json()

        st.write(
            f"**{len(documents_data['documents'])} documents** · "
            f"**{documents_data['total_chunks']} chunks**"
        )

        if not documents_data["documents"]:

            st.info(
                "No documents indexed yet. "
                "Upload one from the sidebar."
            )

        for doc in documents_data["documents"]:

            col1, col2, col3 = st.columns([5, 2, 1])

            with col1:
                st.write(doc["document_name"])

            with col2:
                st.write(f"{doc['chunks']} chunks")

            with col3:

                if st.button(
                    "Delete",
                    key=f"delete_{doc['document_name']}",
                ):

                    delete_response = requests.delete(
                        f"{API_URL}/documents",
                        json={
                            "document_name": doc["document_name"]
                        },
                        timeout=30,
                    )

                    if delete_response.ok:

                        st.success(
                            f"Deleted {doc['document_name']}"
                        )

                        st.rerun()

                    else:

                        st.error(
                            f"Delete failed: {delete_response.text}"
                        )

    else:

        st.error(
            f"Could not load documents: {documents_response.text}"
        )
