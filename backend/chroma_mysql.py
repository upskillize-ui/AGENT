"""
chroma_mysql.py
Replaces ChromaDB disk storage with MySQL-backed vector storage.
Embeddings are stored in Aiven MySQL and never lost on restart.
"""

import os
import json
import hashlib
import numpy as np
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

DATABASE_URL = os.getenv("DATABASE_URL", "")

engine = create_engine(DATABASE_URL, poolclass=NullPool, connect_args={"ssl": {"ca": os.getenv("SSL_CA", "./ca.pem")}})

def init_vector_table():
    """Create the embeddings table if it doesn't exist."""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS rag_embeddings (
                id VARCHAR(64) PRIMARY KEY,
                content_id VARCHAR(64) NOT NULL,
                chunk_index INT DEFAULT 0,
                page_content TEXT NOT NULL,
                embedding JSON NOT NULL,
                lecture_id INT,
                lecture_title VARCHAR(500),
                source_type VARCHAR(50),
                source_title VARCHAR(500),
                source_id VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_content_id (content_id),
                INDEX idx_lecture_id (lecture_id)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """))
        conn.commit()
    print("[MySQL Vector] Table ready.")


def already_indexed(content_id: str) -> bool:
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM rag_embeddings WHERE content_id = :cid"),
            {"cid": content_id}
        )
        return result.scalar() > 0


def add_documents(docs: list, embeddings_model) -> int:
    """Embed and store documents in MySQL."""
    if not docs:
        return 0

    texts = [d.page_content for d in docs]
    vectors = embeddings_model.embed_documents(texts)

    with engine.connect() as conn:
        for doc, vector in zip(docs, vectors):
            m = doc.metadata
            uid = hashlib.md5(
                f"{m.get('content_id','')}_{m.get('chunk_index',0)}".encode()
            ).hexdigest()

            conn.execute(text("""
                INSERT IGNORE INTO rag_embeddings
                (id, content_id, chunk_index, page_content, embedding,
                 lecture_id, lecture_title, source_type, source_title, source_id)
                VALUES
                (:id, :content_id, :chunk_index, :page_content, :embedding,
                 :lecture_id, :lecture_title, :source_type, :source_title, :source_id)
            """), {
                "id": uid,
                "content_id": m.get("content_id", ""),
                "chunk_index": m.get("chunk_index", 0),
                "page_content": doc.page_content,
                "embedding": json.dumps(vector),
                "lecture_id": m.get("lecture_id"),
                "lecture_title": m.get("lecture_title", ""),
                "source_type": m.get("source_type", ""),
                "source_title": m.get("source_title", ""),
                "source_id": m.get("source_id", ""),
            })
        conn.commit()

    return len(docs)


def similarity_search(query_vector: list, lecture_id: Optional[int] = None, top_k: int = 6) -> list:
    """Find most similar chunks using cosine similarity."""
    with engine.connect() as conn:
        if lecture_id:
            rows = conn.execute(
                text("SELECT * FROM rag_embeddings WHERE lecture_id = :lid"),
                {"lid": lecture_id}
            ).fetchall()
        else:
            rows = conn.execute(
                text("SELECT * FROM rag_embeddings")
            ).fetchall()

    if not rows:
        return []

    q = np.array(query_vector)
    scored = []
    for row in rows:
        vec = np.array(json.loads(row.embedding))
        # Cosine similarity
        sim = np.dot(q, vec) / (np.linalg.norm(q) * np.linalg.norm(vec) + 1e-9)
        scored.append((sim, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[:top_k]]