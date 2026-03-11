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
    """
    Generate questions grounded strictly in the retrieved context.
    Returns structured JSON with questions + metadata.
    """
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

    prompt = f"""You are an expert assessment designer for an online learning platform.

Below is the ONLY material you are allowed to use. It has been retrieved from the platform's own lectures, notes, PDFs, and case studies.

=== RETRIEVED COURSE MATERIAL ===
{context}
=== END OF MATERIAL ===

Sources used:
{source_list}

YOUR TASK:
Generate exactly {num_questions} mock test question(s) on the topic: "{topic}"

ABSOLUTE RULES — NEVER BREAK THESE:
1. Every question MUST be directly answerable from the retrieved material above.
2. Avoid using any outside knowledge, general facts, or information not present in the material.
3. If a concept is not in the retrieved material, do NOT create a question about it.
4. Use the exact terminology, examples, and explanations from the source material.
5. The explanation field MUST cite which [SOURCE N] contains the answer and suggested reading.

CONFIGURATION:
- Difficulty: {difficulty} → focus on {DIFFICULTY_GUIDE[difficulty]}
- Question types to include (distribute evenly): {', '.join(question_types)}
- Duration: {duration_minutes} minutes

QUESTION TYPE RULES:
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
      "explanation": "According to [SOURCE 1] (Lecture transcript), the answer is B because ...",
      "source_reference": "[SOURCE 1] — Transcript: Introduction to Neural Networks"
    }}
  ]
}}"""

    message = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    # Strip accidental markdown fences
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
    """
    Evaluate student answers. Feedback is grounded in the source material citations
    already embedded in each question's explanation field.
    """
    qa_breakdown = []
    score = 0

    for q in questions:
        student_ans = answers.get(q["id"], [])
        correct_ans = q.get("correct_answers", [])
        is_correct  = sorted(student_ans) == sorted(correct_ans)
        if is_correct:
            score += 1
        qa_breakdown.append({
            "id":               q["id"],
            "question":         q["question"],
            "type":             q.get("type", "mcq"),
            "options":          q.get("options", {}),
            "student_answer":   student_ans,
            "correct_answer":   correct_ans,
            "is_correct":       is_correct,
            "explanation":      q.get("explanation", ""),
            "source_reference": q.get("source_reference", ""),
        })

    pct = round(score / len(questions) * 100, 1) if questions else 0

    prompt = f"""You are an educational coach reviewing a student's mock test.
All questions were generated from the student's own course materials.

Results: {score}/{len(questions)} correct ({pct}%)
Time taken: {time_taken_seconds // 60}m {time_taken_seconds % 60}s

Detailed breakdown:
{json.dumps(qa_breakdown, indent=2)}

Provide feedback that references the specific source materials (as cited in explanation fields).
Return ONLY valid JSON:
{{
  "score": {score},
  "total": {len(questions)},
  "percentage": {pct},
  "time_taken_seconds": {time_taken_seconds},
  "performance_band": "Excellent|Good|Average|Needs Improvement",
  "overall_feedback": "2-3 sentences referencing the actual course materials covered.",
  "weak_areas": ["concept from material"],
  "strong_areas": ["concept from material"],
  "results": [
    {{
      "id": "q1",
      "is_correct": true,
      "student_answer": ["B"],
      "correct_answer": ["B"],
      "feedback": "Specific feedback citing the source material.",
      "source_reference": "..."
    }}
  ],
  "study_recommendations": [
    "Re-read [specific section] in [source title]",
    "Review [concept] from [lecture title]"
  ]
}}"""

    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
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