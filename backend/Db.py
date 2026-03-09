"""
db.py — MySQL connector
Reads your platform's learning content from MySQL.
Edit the table/column names to match YOUR actual schema.
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from typing import Optional

# ── Connection ────────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:yourpassword@localhost:3306/your_database_name"
)

# Aiven Cloud requires SSL — place ca.pem in the same folder as db.py
_ca_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ca.pem")
_ssl_args = {}
if os.path.exists(_ca_path):
    _ssl_args = {"connect_args": {"ssl": {"ca": _ca_path}}}
    print(f"[DB] SSL enabled using: {_ca_path}")
else:
    print("[DB] WARNING: ca.pem not found — connection may fail on Aiven Cloud")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600, **_ssl_args)
SessionLocal = sessionmaker(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Content fetchers ──────────────────────────────────────────────────────────
# ⚠️  EDIT THESE QUERIES to match your actual MySQL table/column names.
# The function signatures and return shape must stay the same.

def fetch_lecture_content(lecture_id: int) -> dict:
    """
    Fetch a single lecture's text content.
    Returns: { id, title, transcript, description }
    """
    with SessionLocal() as db:
        row = db.execute(text("""
            SELECT id, title, transcript, description
            FROM lectures                          -- ← your table name
            WHERE id = :lecture_id AND is_active = 1
        """), {"lecture_id": lecture_id}).fetchone()

        if not row:
            return {}
        return dict(row._mapping)


def fetch_notes_for_lecture(lecture_id: int) -> list[dict]:
    """
    Fetch all class notes linked to a lecture.
    Returns: [{ id, title, content, format }]
    """
    with SessionLocal() as db:
        rows = db.execute(text("""
            SELECT id, title, content, format    -- format: 'html' | 'text'
            FROM class_notes                     -- ← your table name
            WHERE lecture_id = :lecture_id
            ORDER BY created_at ASC
        """), {"lecture_id": lecture_id}).fetchall()
        return [dict(r._mapping) for r in rows]


def fetch_pdfs_for_lecture(lecture_id: int) -> list[dict]:
    """
    Fetch PDF study materials for a lecture.
    Returns: [{ id, title, file_url, file_path }]
    """
    with SessionLocal() as db:
        rows = db.execute(text("""
            SELECT id, title, file_url, file_path
            FROM study_materials                  -- ← your table name
            WHERE lecture_id = :lecture_id
              AND material_type = 'pdf'
            ORDER BY created_at ASC
        """), {"lecture_id": lecture_id}).fetchall()
        return [dict(r._mapping) for r in rows]


def fetch_case_studies_for_lecture(lecture_id: int) -> list[dict]:
    """
    Fetch case studies linked to a lecture.
    Returns: [{ id, title, body }]
    """
    with SessionLocal() as db:
        rows = db.execute(text("""
            SELECT id, title, body
            FROM case_studies                     -- ← your table name
            WHERE lecture_id = :lecture_id
            ORDER BY created_at ASC
        """), {"lecture_id": lecture_id}).fetchall()
        return [dict(r._mapping) for r in rows]


def fetch_course_content(course_id: int) -> list[dict]:
    """
    Fetch ALL lectures + materials for an entire course.
    Used when student wants a test across the full course.
    Returns: [{ lecture_id, lecture_title, transcript, notes: [], pdfs: [], cases: [] }]
    """
    with SessionLocal() as db:
        lectures = db.execute(text("""
            SELECT id, title, transcript, description
            FROM lectures
            WHERE course_id = :course_id AND is_active = 1
            ORDER BY sequence_order ASC
        """), {"course_id": course_id}).fetchall()

        result = []
        for lec in lectures:
            d = dict(lec._mapping)
            d["notes"]  = fetch_notes_for_lecture(d["id"])
            d["pdfs"]   = fetch_pdfs_for_lecture(d["id"])
            d["cases"]  = fetch_case_studies_for_lecture(d["id"])
            result.append(d)
        return result