"""
rag_pipeline.py
Ingestion  → chunks raw text from MySQL content
Embedding  → sentence-transformers (local, free, no API key needed)
Vector Store → ChromaDB (persistent on disk)
Retriever  → LangChain similarity search
"""

import os
import hashlib
from typing import Optional
import requests

# ── Updated imports — no deprecation warnings ─────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# PDF text extraction
from pypdf import PdfReader
import io

# HTML stripping
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DIR    = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
EMBED_MODEL   = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K         = int(os.getenv("RETRIEVER_TOP_K", "6"))

# ── Embeddings (loaded once at startup) ───────────────────────────────────────

print("[RAG] Loading embedding model:", EMBED_MODEL)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ── Vector store ──────────────────────────────────────────────────────────────

vectorstore = Chroma(
    collection_name="learning_content",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

# ── Text splitter ─────────────────────────────────────────────────────────────

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_html(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)


def _extract_pdf_text(source) -> str:
    """Extract text from a PDF given a URL, local path, or bytes."""
    try:
        if isinstance(source, str) and source.startswith("http"):
            resp = requests.get(source, timeout=30)
            resp.raise_for_status()
            data = io.BytesIO(resp.content)
        elif isinstance(source, (bytes, bytearray)):
            data = io.BytesIO(source)
        else:
            data = open(source, "rb")

        reader = PdfReader(data)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"[RAG] PDF extraction failed: {e}")
        return ""


def _content_id(lecture_id: int, source_type: str, source_id) -> str:
    """Stable ID so we can check if content is already indexed."""
    key = f"{lecture_id}:{source_type}:{source_id}"
    return hashlib.md5(key.encode()).hexdigest()


def _already_indexed(content_id: str) -> bool:
    results = vectorstore.get(where={"content_id": content_id}, limit=1)
    return len(results["ids"]) > 0


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_lecture(lecture_data: dict) -> int:
    lecture_id    = lecture_data["id"]
    lecture_title = lecture_data.get("title", f"Lecture {lecture_id}")
    docs: list[Document] = []

    def add_chunks(text: str, source_type: str, source_id, source_title: str):
        if not text or not text.strip():
            return
        cid = _content_id(lecture_id, source_type, source_id)
        if _already_indexed(cid):
            print(f"[RAG] Already indexed: {source_type} {source_id} — skipping")
            return
        chunks = splitter.split_text(text.strip())
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "lecture_id":    lecture_id,
                    "lecture_title": lecture_title,
                    "source_type":   source_type,
                    "source_title":  source_title,
                    "source_id":     str(source_id),
                    "content_id":    cid,
                    "chunk_index":   i,
                }
            ))

    add_chunks(
        lecture_data.get("transcript", "") or lecture_data.get("description", ""),
        "transcript", lecture_id, f"Transcript — {lecture_title}"
    )

    for note in lecture_data.get("notes", []):
        text = _strip_html(note["content"]) if note.get("format") == "html" else note.get("content", "")
        add_chunks(text, "notes", note["id"], note.get("title", "Class Note"))

    for pdf in lecture_data.get("pdfs", []):
        text = _extract_pdf_text(pdf.get("file_url") or pdf.get("file_path", ""))
        add_chunks(text, "pdf", pdf["id"], pdf.get("title", "PDF Document"))

    for case in lecture_data.get("cases", []):
        add_chunks(case.get("body", ""), "case_study", case["id"], case.get("title", "Case Study"))

    if docs:
        vectorstore.add_documents(docs)
        print(f"[RAG] Ingested {len(docs)} chunks for lecture {lecture_id}")
    return len(docs)


def ingest_course(course_lectures: list[dict]) -> int:
    total = 0
    for lecture in course_lectures:
        total += ingest_lecture(lecture)
    return total


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_context(
    query: str,
    lecture_id: Optional[int] = None,
    course_id: Optional[int] = None,
    top_k: int = TOP_K,
) -> tuple[str, list[dict]]:
    where_filter = None
    if lecture_id:
        where_filter = {"lecture_id": lecture_id}

    results: list[Document] = vectorstore.similarity_search(
        query,
        k=top_k,
        filter=where_filter,
    )

    if not results:
        return "", []

    seen = set()
    unique_docs = []
    for doc in results:
        key = doc.metadata.get("content_id", "") + str(doc.metadata.get("chunk_index", 0))
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    parts = []
    sources = []
    for i, doc in enumerate(unique_docs):
        m = doc.metadata
        header = (
            f"[SOURCE {i+1}] {m.get('source_title', 'Unknown')} "
            f"({m.get('source_type', '').upper()}) | "
            f"Lecture: {m.get('lecture_title', '')}"
        )
        parts.append(f"{header}\n{doc.page_content}")
        sources.append({
            "index":         i + 1,
            "source_title":  m.get("source_title"),
            "source_type":   m.get("source_type"),
            "lecture_title": m.get("lecture_title"),
            "lecture_id":    m.get("lecture_id"),
        })

    return "\n\n---\n\n".join(parts), sources