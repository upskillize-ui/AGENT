"""
question_generator.py
Uses RAG-retrieved context to generate grounded questions via Claude.
Claude ONLY sees your retrieved chunks — no outside knowledge.
"""

import os
import json
import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-haiku-4-5-20251001"

DIFFICULTY_GUIDE = {
    "easy":    "recall and definitions directly stated in the retrieved material",
    "medium":  "application and understanding of concepts from the material",
    "hard":    "analysis, inference, and synthesis across the retrieved chunks",
    "complex": "multi-step reasoning, edge cases, and critical evaluation of the material",
}

TYPE_RULES = {
    "mcq":        "Multiple Choice — exactly ONE correct option from A, B, C, D",
    "msq":        "Multiple Select — TWO OR MORE correct answers from A, B, C, D, E. correct_answers list has 2+ entries.",
    "true_false": 'True/False — options must be exactly {"A": "True", "B": "False"}',
}


def generate_questions(
    topic: str,
    context: str,
    sources: list[dict],
    num_questions: int,
    difficulty: str,
    question_types: list[str],
    duration_minutes: int,
) -> dict:
    if not context.strip():
        raise ValueError(
            "No relevant content found in your platform's materials for this topic. "
            "Please ensure the lecture has been ingested into the vector store."
        )

    types_desc = "\n".join(f"  • {TYPE_RULES[t]}" for t in question_types)
    source_list = "\n".join(
        f"  [{s['index']}] {s['source_title']} ({s['source_type']})"
        for s in sources
    )

    prompt = f"""You are an assessment designer for an online learning platform.

Use ONLY the material below to create questions. No outside knowledge allowed.

=== RETRIEVED COURSE MATERIAL ===
{context}
=== END OF MATERIAL ===

Sources:
{source_list}

Generate exactly {num_questions} question(s) on: "{topic}"

Rules:
1. Every question must be answerable from the material above only.
2. Use exact terminology from the source material.
3. Distribute question types evenly: {', '.join(question_types)}
4. Difficulty: {difficulty} → {DIFFICULTY_GUIDE[difficulty]}

Question type rules:
{types_desc}

Return ONLY valid JSON — no markdown, no extra text:
{{
  "topic": "{topic}",
  "difficulty": "{difficulty}",
  "duration_minutes": {duration_minutes},
  "source_grounded": true,
  "questions": [
    {{
      "id": "q1",
      "type": "mcq",
      "question": "Question text here?",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "correct_answers": ["B"],
      "explanation": "Brief 1-line explanation citing [SOURCE N].",
      "source_reference": "[SOURCE 1] — Transcript: Lecture Title"
    }}
  ]
}}"""

    message = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:])
        if raw.strip().endswith("```"):
            raw = raw.strip()[:-3]

    result = json.loads(raw.strip())
    result["retrieved_sources"] = sources
    return result


def evaluate_answers(
    questions: list[dict],
    answers: dict[str, list[str]],
    student_id: str,
    test_id: str,
    time_taken_seconds: int,
) -> dict:
    qa_breakdown = []
    score = 0

    for q in questions:
        student_ans = answers.get(q["id"], [])
        correct_ans = q.get("correct_answers", [])
        is_correct  = sorted(student_ans) == sorted(correct_ans)
        if is_correct:
            score += 1
        qa_breakdown.append({
            "id":             q["id"],
            "question":       q["question"],
            "type":           q.get("type", "mcq"),
            "student_answer": student_ans,
            "correct_answer": correct_ans,
            "is_correct":     is_correct,
        })

    pct = round(score / len(questions) * 100, 1) if questions else 0

    prompt = f"""You are a friendly tutor giving brief feedback on a student's test.

Score: {score}/{len(questions)} ({pct}%) in {time_taken_seconds // 60}m {time_taken_seconds % 60}s

{json.dumps(qa_breakdown, indent=2)}

Give short, warm, human feedback. Max 1-2 lines per question.
Return ONLY valid JSON:
{{
  "score": {score},
  "total": {len(questions)},
  "percentage": {pct},
  "time_taken_seconds": {time_taken_seconds},
  "performance_band": "Excellent|Good|Average|Needs Improvement",
  "overall_feedback": "One friendly sentence about their performance.",
  "weak_areas": ["topic"],
  "strong_areas": ["topic"],
  "results": [
    {{
      "id": "q1",
      "is_correct": true,
      "student_answer": ["B"],
      "correct_answer": ["B"],
      "feedback": "One short friendly line.",
      "source_reference": ""
    }}
  ],
  "study_recommendations": ["One short tip per weak area, max 3 tips total"]
}}"""

    message = client.messages.create(
        model=MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:])
        if raw.strip().endswith("```"):
            raw = raw.strip()[:-3]

    result = json.loads(raw.strip())
    result["student_id"] = student_id
    result["test_id"]    = test_id
    return result