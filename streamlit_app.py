"""
Agentic RAG - Chat with your Documents
======================================
A self-contained Retrieval-Augmented Generation app:

  upload PDFs  ->  chunk  ->  embed  ->  retrieve  ->  answer WITH citations

Designed to run for free on Streamlit Community Cloud with a
"bring-your-own-key" model: the visitor pastes their own OpenAI API key, so the
public demo costs the author nothing and never exposes a secret.

Run locally:
    pip install -r requirements-streamlit.txt
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import streamlit as st
from pypdf import PdfReader

try:
    from openai import OpenAI
except Exception:  # import guard for first run
    OpenAI = None


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL_DEFAULT = "gpt-4o-mini"
CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 150      # characters of overlap between chunks
TOP_K = 4                # chunks retrieved per question

SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions using ONLY the provided "
    "context passages. Each passage is labelled like [1], [2]. When you use a "
    "passage, cite it inline with its number, e.g. 'The policy expires in 2025 [2].' "
    "If the answer is not contained in the context, say you don't know based on the "
    "documents provided. Never invent facts or citations."
)


@dataclass
class Chunk:
    text: str
    source: str   # file name
    page: int     # 1-based page number


# --------------------------------------------------------------------------- #
# Core RAG helpers
# --------------------------------------------------------------------------- #
def extract_chunks(file_bytes: bytes, filename: str) -> list:
    """Read a PDF and split each page into overlapping character chunks."""
    reader = PdfReader(io.BytesIO(file_bytes))
    chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            piece = text[start : start + CHUNK_SIZE].strip()
            if piece:
                chunks.append(Chunk(text=piece, source=filename, page=page_num))
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed_texts(client, texts: list) -> np.ndarray:
    """Embed a list of texts (batched) and return a normalized (n, d) matrix."""
    vectors = []
    batch = 64
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i : i + batch])
        vectors.extend([d.embedding for d in resp.data])
    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return arr / norms


def retrieve(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list:
    """Return indices of the top-k most similar chunks (cosine similarity)."""
    sims = matrix @ query_vec
    return np.argsort(-sims)[:k].tolist()


def build_context(chunks: list, idxs: list):
    """Assemble the numbered context block and the ordered source list."""
    selected = [chunks[i] for i in idxs]
    blocks = [f"[{n}] {c.text}" for n, c in enumerate(selected, start=1)]
    return "\n\n".join(blocks), selected


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Chat with your Documents", page_icon=":page_facing_up:", layout="wide")
    st.title("Chat with your Documents")
    st.caption("Retrieval-Augmented Generation with inline source citations.")

    ss = st.session_state
    ss.setdefault("chunks", [])
    ss.setdefault("matrix", None)
    ss.setdefault("messages", [])
    ss.setdefault("indexed_files", [])

    with st.sidebar:
        st.header("Setup")
        api_key = st.text_input(
            "OpenAI API key",
            type="password",
            help="Used only in this session; never stored or logged.",
            placeholder="sk-...",
        )
        chat_model = st.selectbox("Chat model", [CHAT_MODEL_DEFAULT, "gpt-4o"], index=0)
        top_k = st.slider("Passages to retrieve", 2, 8, TOP_K)

        st.divider()
        uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
        index_clicked = st.button("Index documents", type="primary", use_container_width=True)

        if ss.indexed_files:
            st.success("Indexed: " + ", ".join(ss.indexed_files))
            st.caption(f"{len(ss.chunks)} chunks ready.")

    if OpenAI is None:
        st.error("The `openai` package isn't installed. Run `pip install -r requirements-streamlit.txt`.")
        return

    if index_clicked:
        if not api_key:
            st.warning("Enter your OpenAI API key first.")
        elif not uploaded:
            st.warning("Upload at least one PDF.")
        else:
            client = OpenAI(api_key=api_key)
            all_chunks = []
            with st.spinner("Reading and chunking PDFs..."):
                for f in uploaded:
                    all_chunks.extend(extract_chunks(f.getvalue(), f.name))
            if not all_chunks:
                st.error("No extractable text found (scanned PDFs need OCR).")
            else:
                try:
                    with st.spinner(f"Embedding {len(all_chunks)} chunks..."):
                        matrix = embed_texts(client, [c.text for c in all_chunks])
                    ss.chunks = all_chunks
                    ss.matrix = matrix
                    ss.indexed_files = [f.name for f in uploaded]
                    ss.messages = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Embedding failed: {e}")

    if ss.matrix is None:
        st.info("In the sidebar: add your API key, upload a PDF, and click **Index documents** to begin.")
        return

    for msg in ss.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for n, c in enumerate(msg["sources"], start=1):
                        st.markdown(f"**[{n}]** `{c.source}` - page {c.page}")
                        st.caption(c.text[:300] + ("..." if len(c.text) > 300 else ""))

    question = st.chat_input("Ask a question about your documents...")
    if not question:
        return
    if not api_key:
        st.warning("Enter your OpenAI API key in the sidebar.")
        return

    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    client = OpenAI(api_key=api_key)
    try:
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and reasoning..."):
                qvec = embed_texts(client, [question])[0]
                idxs = retrieve(qvec, ss.matrix, top_k)
                context, sources = build_context(ss.chunks, idxs)
                resp = client.chat.completions.create(
                    model=chat_model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                    ],
                )
                answer = resp.choices[0].message.content
            st.markdown(answer)
            with st.expander("Sources"):
                for n, c in enumerate(sources, start=1):
                    st.markdown(f"**[{n}]** `{c.source}` - page {c.page}")
                    st.caption(c.text[:300] + ("..." if len(c.text) > 300 else ""))
        ss.messages.append({"role": "assistant", "content": answer, "sources": sources})
    except Exception as e:
        st.error(f"Request failed: {e}")


if __name__ == "__main__":
    main()
