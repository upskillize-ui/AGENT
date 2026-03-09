"""
main.py — RAG-Based AI Mock Test Agent
FastAPI + ChromaDB + sentence-transformers + Claude
"""

import os
import uuid
import time
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from Db import (
    fetch_lecture_content,
    fetch_notes_for_lecture,
    fetch_pdfs_for_lecture,
    fetch_case_studies_for_lecture,
    fetch_course_content,
)
from rag_pipeline import ingest_lecture, ingest_course, retrieve_context
from questions_generator import generate_questions, evaluate_answers

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Mock Test Agent (RAG)",
    description="Generates questions exclusively from your platform's content using RAG.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        os.getenv("FRONTEND_URL", "https://lms.upskillize.com"),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────────────────────

class IngestLectureRequest(BaseModel):
    lecture_id: int
    force_reingest: bool = False   # set True to re-embed even if already indexed

class IngestCourseRequest(BaseModel):
    course_id: int
    force_reingest: bool = False

class GenerateTestRequest(BaseModel):
    topic: str = Field(..., description="Topic or subject to test on")
    num_questions: int = Field(default=10, ge=1, le=50)
    duration_minutes: int = Field(default=30, ge=5, le=180)
    difficulty: Literal["easy", "medium", "hard", "complex"] = "medium"
    question_types: list[Literal["mcq", "msq", "true_false"]] = ["mcq"]
    student_id: Optional[str] = None
    # Scope — at least one required
    lecture_id: Optional[int] = None   # test from a single lecture
    course_id: Optional[int] = None    # test from the whole course

class SubmitAnswersRequest(BaseModel):
    test_id: str
    student_id: str
    questions: list[dict]
    answers: dict[str, list[str]]      # {"q1": ["B"], "q2": ["A","C"]}
    time_taken_seconds: int

# ── Background ingestion task ─────────────────────────────────────────────────

def _ingest_lecture_task(lecture_id: int):
    """Runs in background after trigger — safe to call from Node.js webhook."""
    lecture = fetch_lecture_content(lecture_id)
    if not lecture:
        print(f"[INGEST] Lecture {lecture_id} not found in DB")
        return
    lecture["notes"] = fetch_notes_for_lecture(lecture_id)
    lecture["pdfs"]  = fetch_pdfs_for_lecture(lecture_id)
    lecture["cases"] = fetch_case_studies_for_lecture(lecture_id)
    count = ingest_lecture(lecture)
    print(f"[INGEST] Lecture {lecture_id} — {count} new chunks added")

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "agent": "RAG Mock Test Agent",
        "version": "3.0.0",
        "embed_model": os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"),
    }


@app.post("/api/ingest/lecture")
async def ingest_lecture_endpoint(
    req: IngestLectureRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger ingestion of a single lecture into the vector store.
    Call this from Node.js when a lecture is created/updated.
    Runs in the background — returns immediately.
    """
    background_tasks.add_task(_ingest_lecture_task, req.lecture_id)
    return {
        "status": "ingestion_started",
        "lecture_id": req.lecture_id,
        "message": "Content is being embedded in the background.",
    }


@app.post("/api/ingest/course")
async def ingest_course_endpoint(
    req: IngestCourseRequest,
    background_tasks: BackgroundTasks,
):
    """Ingest all lectures for an entire course in the background."""
    def task():
        lectures = fetch_course_content(req.course_id)
        total = ingest_course(lectures)
        print(f"[INGEST] Course {req.course_id} — {total} total chunks added")

    background_tasks.add_task(task)
    return {
        "status": "ingestion_started",
        "course_id": req.course_id,
        "message": "All course content is being embedded in the background.",
    }


@app.post("/api/generate-test")
async def generate_test(req: GenerateTestRequest):
    """
    RAG pipeline:
      1. Semantic search over your vectorised content
      2. Retrieved chunks → Claude prompt (strictly grounded)
      3. Returns structured test JSON
    """
    if not req.lecture_id and not req.course_id:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: lecture_id or course_id"
        )

    # Step 1 — Retrieve relevant chunks
    context, sources = retrieve_context(
        query=req.topic,
        lecture_id=req.lecture_id,
    )

    if not context:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No content found for topic '{req.topic}'. "
                "Make sure the lecture has been ingested via POST /api/ingest/lecture."
            )
        )

    # Step 2 — Generate questions grounded in retrieved context
    try:
        test_data = generate_questions(
            topic=req.topic,
            context=context,
            sources=sources,
            num_questions=req.num_questions,
            difficulty=req.difficulty,
            question_types=req.question_types,
            duration_minutes=req.duration_minutes,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    test_data["test_id"]       = f"test_{uuid.uuid4().hex[:10]}"
    test_data["generated_at"]  = int(time.time())
    test_data["lecture_id"]    = req.lecture_id
    test_data["student_id"]    = req.student_id
    return test_data


@app.post("/api/submit-answers")
async def submit_answers(req: SubmitAnswersRequest):
    """Evaluate answers and return grounded feedback + study recommendations."""
    try:
        result = evaluate_answers(
            questions=req.questions,
            answers=req.answers,
            student_id=req.student_id,
            test_id=req.test_id,
            time_taken_seconds=req.time_taken_seconds,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")