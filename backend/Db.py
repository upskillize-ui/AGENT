"""
db.py — MySQL connector for upskillize_lms
Mapped to actual schema: lessons, course_modules, courses
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:yourpassword@localhost:3306/upskillize_lms"
)

_ca_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ca.pem")
_ssl_args = {}
if os.path.exists(_ca_path):
    _ssl_args = {"connect_args": {"ssl": {"ca": _ca_path}}}
    print(f"[DB] SSL enabled using: {_ca_path}")
else:
    print("[DB] WARNING: ca.pem not found")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600, **_ssl_args)
SessionLocal = sessionmaker(bind=engine)


def fetch_lecture_content(lecture_id: int) -> dict:
    """Fetch a single lesson by ID."""
    with SessionLocal() as db:
        row = db.execute(text("""
            SELECT l.id,
                   l.lesson_name   AS title,
                   l.description   AS transcript,
                   l.description,
                   l.content_type,
                   l.content_url,
                   l.youtube_video_id,
                   c.course_name,
                   cm.module_name
            FROM lessons l
            JOIN course_modules cm ON l.course_module_id = cm.id
            JOIN courses c         ON cm.course_id = c.id
            WHERE l.id = :lecture_id
        """), {"lecture_id": lecture_id}).fetchone()

        if not row:
            return {}
        return dict(row._mapping)


def fetch_notes_for_lecture(lecture_id: int) -> list[dict]:
    """No notes table in schema — returns empty list."""
    return []


def fetch_pdfs_for_lecture(lecture_id: int) -> list[dict]:
    """Fetch PDF/PPT lessons linked to the same module as this lesson."""
    with SessionLocal() as db:
        rows = db.execute(text("""
            SELECT l.id, l.lesson_name AS title, l.content_url AS file_url
            FROM lessons l
            WHERE l.course_module_id = (
                SELECT course_module_id FROM lessons WHERE id = :lecture_id
            )
            AND l.content_type IN ('pdf', 'ppt')
        """), {"lecture_id": lecture_id}).fetchall()
        return [dict(r._mapping) for r in rows]


def fetch_case_studies_for_lecture(lecture_id: int) -> list[dict]:
    """No case_studies table in schema — returns empty list."""
    return []


def fetch_course_content(course_id: int) -> list[dict]:
    """
    Fetch ALL lessons for an entire course across all modules.
    This is what gets called when student ingests a course for TestGen.
    """
    with SessionLocal() as db:
        rows = db.execute(text("""
            SELECT
                l.id,
                l.lesson_name        AS title,
                l.description        AS transcript,
                l.content_type,
                l.content_url,
                l.youtube_video_id,
                cm.module_name,
                c.course_name,
                c.id                 AS course_id
            FROM lessons l
            JOIN course_modules cm ON l.course_module_id = cm.id
            JOIN courses c         ON cm.course_id = c.id
            WHERE c.id = :course_id
            ORDER BY cm.sequence_order ASC, l.sequence_order ASC
        """), {"course_id": course_id}).fetchall()

        result = []
        for row in rows:
            d = dict(row._mapping)
            d["notes"] = []
            d["pdfs"]  = []
            d["cases"] = []
            result.append(d)
        return result