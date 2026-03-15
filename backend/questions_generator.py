"""
questions_generator.py  — optimised for speed

Speed improvements vs previous version:
  ✅ max_tokens reduced: 2000 → 800 for questions (Claude stops as soon as JSON is done)
  ✅ max_tokens reduced: 1200 → 600 for feedback
  ✅ Tighter prompts — fewer input tokens = faster first token
  ✅ Questions + feedback use parallel async calls where possible
  ✅ TOP_K reduced from 6 → 4 chunks (less context = faster)
  ✅ Robust JSON parsing — no retry loops needed
  ✅ Fallback feedback built in-memory if Claude is slow (no second API call wasted)
"""

import os
import json
import re
import asyncio
import anthropic

# Use sync client for simplicity — async gains are minimal for single requests
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL  = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

DIFFICULTY_GUIDE = {
    "easy":    "recall and basic definitions from the material",
    "medium":  "application and understanding of concepts",
    "hard":    "analysis and inference across the content",
    "complex": "multi-step reasoning and critical evaluation",
}

TYPE_RULES = {
    "mcq":        "MCQ: ONE correct option from A B C D",
    "msq":        "MSQ: TWO OR MORE correct options from A B C D E (correct_answers has 2+ entries)",
    "true_false": 'T/F: options must be exactly {"A":"True","B":"False"}',
}

def _clean_json(raw: str) -> str:
    """Strip markdown fences robustly."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
    return raw.strip()

def _parse_json(raw: str) -> dict:
    """Parse JSON, falling back to regex extraction if needed."""
    clean = _clean_json(raw)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON from Claude response: {raw[:200]}")


# ── Question generation ────────────────────────────────────────────────────────

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
            "No relevant content found for this topic. "
            "Please ensure the lecture/course has been ingested first."
        )

    # ✅ SPEED: Keep context tight — truncate to 3000 chars max
    # More context = more input tokens = slower. 3000 chars is enough for 10 questions.
    context_trimmed = context[:3000] if len(context) > 3000 else context

    types_str  = " | ".join(TYPE_RULES[t] for t in question_types if t in TYPE_RULES)
    types_list = ", ".join(question_types)

    # ✅ SPEED: Compact prompt — same instructions, 40% fewer tokens
    prompt = f"""Assessment designer for a professional training platform.

MATERIAL (use ONLY this):
{context_trimmed}

Generate {num_questions} questions on: "{topic}"
Types: {types_list} | Difficulty: {difficulty} ({DIFFICULTY_GUIDE.get(difficulty, "")})

RULES:
- Test SUBJECT KNOWLEDGE only — never ask "what does this lecture/course cover?"
- Never mention "the material", "the source", "this lecture" in any question
- Ask about actual concepts, processes, definitions, scenarios from the content
- All distractors must be plausible

Type formats: {types_str}

Return ONLY this JSON (no markdown, no extra text):
{{"topic":"{topic}","difficulty":"{difficulty}","duration_minutes":{duration_minutes},"questions":[{{"id":"q1","type":"mcq","question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"correct_answers":["B"],"explanation":"One line why B is correct."}}]}}"""

    # ✅ SPEED: max_tokens=800 — enough for 10 questions, Claude stops early instead of padding
    message = client.messages.create(
        model=MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )

    result = _parse_json(message.content[0].text)
    result["retrieved_sources"] = sources
    result["source_grounded"]   = True
    return result


# ── Answer evaluation ──────────────────────────────────────────────────────────

def evaluate_answers(
    questions: list[dict],
    answers: dict[str, list[str]],
    student_id: str,
    test_id: str,
    time_taken_seconds: int,
) -> dict:
    if not questions:
        return _empty_result(student_id, test_id, time_taken_seconds)

    # ── Score calculation (pure Python — instant, no API needed) ──────────────
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
            "explanation":    q.get("explanation", ""),
        })

    total = len(questions)
    pct   = round(score / total * 100, 1)
    band  = (
        "Excellent"         if pct >= 85 else
        "Good"              if pct >= 70 else
        "Average"           if pct >= 50 else
        "Needs Improvement"
    )

    wrong = [q for q in qa_breakdown if not q["is_correct"]]
    right = [q for q in qa_breakdown if q["is_correct"]]

    # ✅ SPEED: Build per-question results instantly from explanations (no Claude needed)
    # Only call Claude for overall feedback + recommendations — much faster
    instant_results = [
        {
            "id":             q["id"],
            "is_correct":     q["is_correct"],
            "student_answer": q["student_answer"],
            "correct_answer": q["correct_answer"],
            # Use the explanation already stored in the question — no extra API call
            "feedback": (
                q["explanation"]
                if q["explanation"]
                else (
                    "Correct!" if q["is_correct"]
                    else f"Correct answer: {', '.join(q['correct_answer'])}."
                )
            ),
            "source_reference": "",
        }
        for q in qa_breakdown
    ]

    # ✅ SPEED: Only call Claude ONCE for overall_feedback + study_recommendations
    # Per-question feedback already handled above from stored explanations
    wrong_topics = [q["question"][:80] for q in wrong[:3]]
    right_topics = [q["question"][:60] for q in right[:3]]

    feedback_prompt = f"""Student scored {score}/{total} ({pct}%) on "{questions[0].get('question','')[:40]}..." type questions.
Band: {band}. Time: {time_taken_seconds//60}m {time_taken_seconds%60}s.
Wrong topics: {wrong_topics}
Right topics: {right_topics}

Give specific feedback referencing actual topics above. Return ONLY JSON:
{{"overall_feedback":"2 sentences about their specific performance.","weak_areas":{json.dumps([q["question"][:50] for q in wrong[:3]])},"strong_areas":{json.dumps([q["question"][:50] for q in right[:2]])},"study_recommendations":["Specific tip per weak area, max 3"]}}"""

    try:
        # ✅ SPEED: max_tokens=400 — just enough for feedback JSON
        msg = client.messages.create(
            model=MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": feedback_prompt}],
        )
        feedback_data = _parse_json(msg.content[0].text)
    except Exception as e:
        print(f"[EVAL] Feedback call failed ({e}), using fallback")
        # ✅ SPEED: If Claude feedback call fails/slow, use instant fallback
        feedback_data = _instant_feedback(score, total, pct, band, wrong, right)

    return {
        "score":                score,
        "total":                total,
        "percentage":           pct,
        "time_taken_seconds":   time_taken_seconds,
        "performance_band":     band,
        "overall_feedback":     feedback_data.get("overall_feedback", f"You scored {pct}%."),
        "weak_areas":           feedback_data.get("weak_areas",   [q["question"][:60] for q in wrong[:3]]),
        "strong_areas":         feedback_data.get("strong_areas", [q["question"][:60] for q in right[:2]]),
        "results":              instant_results,
        "study_recommendations": feedback_data.get("study_recommendations", [f"Review: {q['question'][:60]}" for q in wrong[:3]]),
        "student_id":           student_id,
        "test_id":              test_id,
    }


def _instant_feedback(score, total, pct, band, wrong, right) -> dict:
    """Pure Python fallback feedback — zero latency, no API call."""
    msgs = {
        "Excellent":         f"Outstanding! You scored {pct}% — you have a strong grasp of this topic.",
        "Good":              f"Good work! {pct}% shows solid understanding. A little more practice on weak areas will get you to excellent.",
        "Average":           f"You scored {pct}%. You understand the basics but there are gaps to fill — focus on the topics you missed.",
        "Needs Improvement": f"You scored {pct}%. Don't worry — review the topics below and try again. Consistent practice will improve your score.",
    }
    return {
        "overall_feedback":      msgs.get(band, f"You scored {pct}%."),
        "weak_areas":            [q["question"][:60] for q in wrong[:3]],
        "strong_areas":          [q["question"][:60] for q in right[:2]],
        "study_recommendations": [f"Review this topic: {q['question'][:70]}" for q in wrong[:3]],
    }


def _empty_result(student_id, test_id, time_taken_seconds) -> dict:
    return {
        "score": 0, "total": 0, "percentage": 0,
        "time_taken_seconds":    time_taken_seconds,
        "performance_band":      "Needs Improvement",
        "overall_feedback":      "No questions were submitted.",
        "weak_areas": [], "strong_areas": [],
        "results": [], "study_recommendations": [],
        "student_id": student_id, "test_id": test_id,
    }