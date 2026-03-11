"""
rag_pipeline.py
Ingestion  → chunks raw text from MySQL content
Embedding  → sentence-transformers (local, free)
Vector Store → Aiven MySQL (persistent, never lost on restart)
Retriever  → cosine similarity search
"""

import os
import hashlib
from typing import Optional
import requests

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from pypdf import PdfReader
import io
from bs4 import BeautifulSoup

from chroma_mysql import init_vector_table, already_indexed, add_documents, similarity_search

# ── Config ────────────────────────────────────────────────────────────────────

EMBED_MODEL   = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K         = int(os.getenv("RETRIEVER_TOP_K", "6"))

# ── Embeddings ────────────────────────────────────────────────────────────────

print("[RAG] Loading embedding model:", EMBED_MODEL)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ── Init MySQL vector table ───────────────────────────────────────────────────

init_vector_table()

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
    key = f"{lecture_id}:{source_type}:{source_id}"
    return hashlib.md5(key.encode()).hexdigest()


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_lecture(lecture_data: dict) -> int:
    lecture_id    = lecture_data["id"]
    lecture_title = lecture_data.get("title", f"Lecture {lecture_id}")
    docs: list[Document] = []

    def add_chunks(text: str, source_type: str, source_id, source_title: str):
        if not text or not text.strip():
            return
        cid = _content_id(lecture_id, source_type, source_id)
        if already_indexed(cid):
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
        count = add_documents(docs, embeddings)
        print(f"[RAG] Ingested {count} chunks for lecture {lecture_id}")
        return count
    return 0


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

    query_vector = embeddings.embed_query(query)
    results = similarity_search(query_vector, lecture_id=lecture_id, top_k=top_k)

    if not results:
        return "", []

    parts = []
    sources = []
    seen = set()

    for i, row in enumerate(results):
        key = f"{row.content_id}_{row.chunk_index}"
        if key in seen:
            continue
        seen.add(key)

        header = (
            f"[SOURCE {i+1}] {row.source_title} "
            f"({row.source_type.upper()}) | "
            f"Lecture: {row.lecture_title}"
        )
        parts.append(f"{header}\n{row.page_content}")
        sources.append({
            "index":         i + 1,
            "source_title":  row.source_title,
            "source_type":   row.source_type,
            "lecture_title": row.lecture_title,
            "lecture_id":    row.lecture_id,
        })

    return "\n\n---\n\n".join(parts), sources