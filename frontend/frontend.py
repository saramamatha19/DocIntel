import html
import re

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


# ============================================================
# Inline citation hover-popup styling
# Injected once per page load. Pure CSS: `.citation-popup` is
# hidden by default and shown on hover/focus of its parent
# `.citation` span, so no server round-trip or Streamlit rerun
# happens on hover.
# ============================================================

st.markdown(
    """
    <style>
    .citation {
        position: relative;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background-color: #E7EDF6;
        color: #2A4B7C;
        font-size: 11px;
        font-weight: 700;
        cursor: pointer;
        margin: 0 2px;
        vertical-align: super;
    }
    .citation-popup {
        display: none;
        position: absolute;
        bottom: 130%;
        left: 0;
        z-index: 999;
        width: 320px;
        max-height: 220px;
        overflow-y: auto;
        background: #ffffff;
        color: #171B24;
        border: 1px solid #DCE0DA;
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 13px;
        font-weight: 400;
        line-height: 1.4;
        text-align: left;
        white-space: normal;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
    }
    .citation:hover .citation-popup,
    .citation:focus .citation-popup {
        display: block;
    }
    .confidence-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 10px 0 4px;
    }
    .confidence-label {
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        color: #8A93A3;
        font-size: 11px;
    }
    .confidence-track {
        width: 120px;
        height: 6px;
        border-radius: 4px;
        background: #EDEEEA;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
    }
    .confidence-fill.good { background: #1F8A5C; }
    .confidence-fill.warn { background: #B7791F; }
    .confidence-fill.critical { background: #C0433A; }
    .confidence-pct {
        font-weight: 700;
        font-size: 12px;
    }
    .confidence-pct.good { color: #1F8A5C; }
    .confidence-pct.warn { color: #B7791F; }
    .confidence-pct.critical { color: #C0433A; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Render the confidence meter for one answer, color-coded by
# band (good/warn/critical) so low-confidence answers are
# visibly flagged rather than looking the same as a strong one.
# ============================================================

def render_confidence(confidence):

    band = confidence["band"]
    percent = confidence["percent"]

    st.markdown(
        f'<div class="confidence-row">'
        f'<span class="confidence-label">Retrieval Confidence</span>'
        f'<div class="confidence-track">'
        f'<div class="confidence-fill {band}" '
        f'style="width:{percent}%"></div>'
        f'</div>'
        f'<span class="confidence-pct {band}">{percent}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# Document category badge. Adding a category later is a
# one-line addition here — must stay in sync with
# DOCUMENT_CATEGORIES in backend/call_llm.py.
# ============================================================

CATEGORY_COLORS = {
    "HR": "#7A4FB0",
    "Legal & Compliance": "#2A4B7C",
    "Security & IT": "#A85C2E",
    "Engineering & Product": "#00968C",
    "Finance": "#1F8A5C",
    "Sales & Marketing": "#C23B7A",
    "Operations & Facilities": "#8A6D3B",
    "Executive & Strategy": "#4B5468",
    "Other": "#8A93A3",
}


def render_category_badge(category):

    color = CATEGORY_COLORS.get(
        category, CATEGORY_COLORS["Other"]
    )

    st.markdown(
        f'<span style="background:{color}1A; color:{color}; '
        f'padding:3px 10px; border-radius:6px; '
        f'font-size:0.75rem; font-weight:700; '
        f'white-space:nowrap;">{category}</span>',
        unsafe_allow_html=True,
    )


# ============================================================
# Turn a [1], [2]... citation marker in the LLM's answer into
# an inline hover-popup span carrying that source's document,
# page, and quoted text. Falls back to leaving the marker as
# plain text if the LLM ever cites a number we don't have.
# ============================================================

def render_answer_with_citations(answer_text, sources):

    def replace_marker(match):

        index = int(match.group(1))

        if index < 1 or index > len(sources):
            return match.group(0)

        source = sources[index - 1]

        if "url" in source:
            location = source["url"]
        else:
            location = (
                f"{source['document_name']} "
                f"(Page {source['page_number']})"
            )

        # Blank lines inside an inline HTML tag get treated by
        # the markdown renderer as a paragraph break, which
        # splits the content out of this (hidden) span and
        # spills it into the page as visible text. Converting
        # newlines to <br> keeps it one unbroken inline string.
        quoted_text = html.escape(
            source["text"]
        ).replace("\n", "<br>")

        popup_html = (
            '<span class="citation-popup">'
            f"<strong>{html.escape(location)}</strong><br>"
            f'"{quoted_text}"'
            "</span>"
        )

        return (
            f'<span class="citation" tabindex="0">{index}'
            f"{popup_html}"
            "</span>"
        )

    return re.sub(r"\[(\d+)\]", replace_marker, answer_text)


# Header
st.title("📚 DocIntel")

st.write(
    "AI-powered document intelligence and RAG assistant"
)

# Summaries of documents uploaded this session, shown at the
# top of the Chat tab (not the Documents tab) so they're the
# first thing you see right after uploading, not something you
# have to go find in a separate list.
if "recent_uploads" not in st.session_state:
    st.session_state.recent_uploads = []


# Sidebar - Document Upload

st.sidebar.header("📄 Add Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)


if st.sidebar.button("Upload & Index"):

    if not uploaded_files:

        st.sidebar.warning(
            "Please select at least one PDF first."
        )

    else:

        upload_results = []

        with st.sidebar.spinner(
            f"Uploading and indexing "
            f"{len(uploaded_files)} file(s)..."
        ):

            for uploaded_file in uploaded_files:

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

                        upload_results.append({
                            "filename": uploaded_file.name,
                            "status": "duplicate",
                            "detail": data["message"],
                        })

                    else:

                        upload_results.append({
                            "filename": uploaded_file.name,
                            "status": "indexed",
                            "detail": (
                                f"{data['chunks_stored']} chunks"
                            ),
                        })

                        st.session_state.recent_uploads.append({
                            "filename": uploaded_file.name,
                            "summary": data.get("summary", ""),
                        })

                else:

                    upload_results.append({
                        "filename": uploaded_file.name,
                        "status": "failed",
                        "detail": response.text,
                    })


        for result in upload_results:

            if result["status"] == "indexed":

                st.sidebar.success(
                    f"✅ {result['filename']} — "
                    f"{result['detail']}"
                )

            elif result["status"] == "duplicate":

                st.sidebar.warning(
                    f"⚠️ {result['filename']} — "
                    f"{result['detail']}"
                )

            else:

                st.sidebar.error(
                    f"❌ {result['filename']} — "
                    f"{result['detail']}"
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

                st.session_state.recent_uploads.append({
                    "filename": data["filename"],
                    "summary": data.get("summary", ""),
                })

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

    # Show what's been uploaded this session, front and center,
    # before you'd even think to ask a question about it.
    for upload in st.session_state.recent_uploads:

        st.info(
            f"📄 **{upload['filename']}** — {upload['summary']}"
        )


    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


    # Render previous turns (answers + sources)

    for turn in st.session_state.chat_history:

        with st.chat_message("user"):

            st.write(
                turn["question"]
            )

        with st.chat_message("assistant"):

            st.markdown(
                render_answer_with_citations(
                    turn["answer"],
                    turn["sources"],
                ),
                unsafe_allow_html=True,
            )

            render_confidence(turn["confidence"])

            if turn["sources"]:

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

                        quoted_text = source["text"].replace(
                            "\n", "\n> "
                        )

                        st.markdown(
                            f"> {quoted_text}"
                        )

                        st.divider()

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
                    "confidence": data["confidence"],
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

            col1, col2, col3, col4 = st.columns([4, 2, 2, 1])

            with col1:
                st.write(doc["document_name"])

            with col2:
                render_category_badge(
                    doc.get("category", "Other")
                )

            with col3:
                st.write(f"{doc['chunks']} chunks")

            with col4:

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
