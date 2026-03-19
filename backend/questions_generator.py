"""
questions_generator.py — FIXED VERSION

BUG FIX: max_tokens was 600 — too low for 10 questions in JSON.
         10 questions need ~2500-4000 tokens. Truncated JSON = parse error = 500.

FIXES APPLIED:
✅ FIX 1: max_tokens scaled by num_questions (250 tokens per question)
✅ FIX 2: Evaluation feedback max_tokens increased to 500
✅ FIX 3: Better JSON parsing with fallback
✅ FIX 4: Proper error messages instead of generic 500
"""

import os
import json
import re
import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

DIFFICULTY_GUIDE = {
    "easy": "recall and basic definitions from the material",
    "medium": "application and understanding of concepts",
    "hard": "analysis and inference across the content",
    "complex": "multi-step reasoning and critical evaluation",
}

TYPE_RULES = {
    "mcq": "MCQ: ONE correct option from A B C D",
    "msq": "MSQ: TWO OR MORE correct options from A B C D E",
    "true_false": 'T/F: options must be {"A":"True","B":"False"}',
}


def _clean_json(raw: str) -> str:
    """Strip markdown fences."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
    return raw.strip()


def _parse_json(raw: str) -> dict:
    """Parse JSON robustly with multiple fallback strategies."""
    clean = _clean_json(raw)
    
    # Strategy 1: Direct parse
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Find JSON object
    try:
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Fix truncated JSON (common when max_tokens is hit)
    try:
        # Try to close any unclosed brackets/braces
        fixed = clean
        open_braces = fixed.count('{') - fixed.count('}')
        open_brackets = fixed.count('[') - fixed.count(']')
        
        # Remove last incomplete item if needed
        if open_braces > 0 or open_brackets > 0:
            # Find last complete question object
            last_complete = fixed.rfind('"explanation"')
            if last_complete > 0:
                # Find the end of that explanation value
                after = fixed[last_complete:]
                quote_end = after.find('"}')
                if quote_end > 0:
                    fixed = fixed[:last_complete + quote_end + 2]
            
            # Close remaining brackets
            fixed += ']' * max(0, open_brackets)
            fixed += '}' * max(0, open_braces)
            
            return json.loads(fixed)
    except (json.JSONDecodeError, Exception):
        pass
    
    raise ValueError(f"Could not parse JSON from AI response (length: {len(raw)} chars). Response may have been truncated.")


# ============================================================================
# QUESTION GENERATION — FIXED max_tokens
# ============================================================================

def generate_questions(
    topic: str,
    context: str,
    sources: list,
    num_questions: int,
    difficulty: str,
    question_types: list,
    duration_minutes: int,
) -> dict:
    """Generate questions with proper token allocation."""
    
    if not context.strip():
        raise ValueError("No relevant content found for this topic.")

    # Truncate context (balance between quality and speed)
    max_context = min(4000, len(context))
    context_trimmed = context[:max_context]

    types_str = " | ".join(TYPE_RULES.get(t, t) for t in question_types)
    types_list = ", ".join(question_types)

    prompt = f"""Generate {num_questions} exam questions on: "{topic}"
Types: {types_list} | Difficulty: {difficulty}
Material (use ONLY this):
{context_trimmed}

RULES:
- Test subject knowledge only
- Never mention "material", "lecture", "source"
- All answers must be in the material above

Types: {types_str}

Return ONLY JSON (no markdown):
{{"topic":"{topic}","difficulty":"{difficulty}","duration_minutes":{duration_minutes},"questions":[{{"id":"q1","type":"mcq","question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"correct_answers":["B"],"explanation":"Why B is correct."}}]}}"""

    # ✅ FIX 1: Scale max_tokens based on number of questions
    # Each question needs ~250 tokens (question + 4 options + explanation)
    # Plus ~200 tokens for the JSON wrapper
    tokens_needed = 200 + (num_questions * 280)
    max_tokens = min(max(tokens_needed, 800), 4096)
    
    print(f"[QGen] Generating {num_questions} questions, max_tokens={max_tokens}")

    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        
        raw_text = message.content[0].text
        
        # Check if response was truncated
        if message.stop_reason == "max_tokens":
            print(f"[QGen] ⚠️ Response was truncated (hit {max_tokens} token limit). Attempting recovery...")
        
        result = _parse_json(raw_text)
        
    except ValueError as e:
        # JSON parsing failed
        print(f"[QGen] ❌ JSON parse failed: {e}")
        raise ValueError(f"Failed to generate valid questions. Please try again with fewer questions (try {max(1, num_questions // 2)}).")
    except anthropic.APIError as e:
        print(f"[QGen] ❌ Anthropic API error: {e}")
        raise ValueError(f"AI service error: {str(e)[:100]}. Please try again.")
    except Exception as e:
        print(f"[QGen] ❌ Unexpected error: {e}")
        raise ValueError(f"Question generation failed: {str(e)[:100]}")

    result["retrieved_sources"] = sources
    result["source_grounded"] = True
    return result


# ============================================================================
# ANSWER EVALUATION — FIXED
# ============================================================================

def evaluate_answers(
    questions: list,
    answers: dict,
    student_id: str,
    test_id: str,
    time_taken_seconds: int,
) -> dict:
    """Evaluate answers with proper error handling."""
    
    if not questions:
        return _empty_result(student_id, test_id, time_taken_seconds)

    # Score in pure Python (instant)
    qa_breakdown = []
    score = 0

    for q in questions:
        student_ans = answers.get(q["id"], [])
        correct_ans = q.get("correct_answers", [])
        is_correct = sorted(student_ans) == sorted(correct_ans)
        if is_correct:
            score += 1
        qa_breakdown.append({
            "id": q["id"],
            "question": q["question"],
            "type": q.get("type", "mcq"),
            "student_answer": student_ans,
            "correct_answer": correct_ans,
            "is_correct": is_correct,
            "explanation": q.get("explanation", ""),
        })

    total = len(questions)
    pct = round(score / total * 100, 1) if total > 0 else 0
    band = (
        "Excellent" if pct >= 85
        else "Good" if pct >= 70
        else "Average" if pct >= 50
        else "Needs Improvement"
    )

    wrong = [q for q in qa_breakdown if not q["is_correct"]]

    # Per-question results using stored explanations (instant)
    instant_results = [
        {
            "id": q["id"],
            "is_correct": q["is_correct"],
            "student_answer": q["student_answer"],
            "correct_answer": q["correct_answer"],
            "feedback": (
                q["explanation"]
                if q["explanation"]
                else ("Correct!" if q["is_correct"] else f"Correct: {', '.join(q['correct_answer'])}")
            ),
        }
        for q in qa_breakdown
    ]

    # Overall feedback from Claude (one call)
    wrong_topics = [q["question"][:80] for q in wrong[:3]]

    feedback_prompt = f"""Student scored {score}/{total} ({pct}%) on test. Band: {band}.
Wrong topics: {wrong_topics}

Give specific feedback. Return ONLY JSON:
{{"overall_feedback":"2 sentences about their performance.","study_recommendations":["Tip 1","Tip 2"]}}"""

    try:
        # ✅ FIX 2: Increased from 300 to 500
        msg = client.messages.create(
            model=MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": feedback_prompt}],
        )
        feedback_data = _parse_json(msg.content[0].text)
    except Exception as e:
        print(f"[EVAL] Feedback call failed ({e}), using fallback")
        feedback_data = _instant_feedback(score, total, pct, band, wrong)

    return {
        "score": score,
        "total": total,
        "percentage": pct,
        "time_taken_seconds": time_taken_seconds,
        "performance_band": band,
        "overall_feedback": feedback_data.get("overall_feedback", f"You scored {pct}%."),
        "results": instant_results,
        "study_recommendations": feedback_data.get(
            "study_recommendations",
            [f"Review: {q['question'][:60]}" for q in wrong[:3]],
        ),
        "student_id": student_id,
        "test_id": test_id,
    }


def _instant_feedback(score, total, pct, band, wrong) -> dict:
    """Fallback feedback (zero latency)."""
    msgs = {
        "Excellent": f"Outstanding! You scored {pct}%.",
        "Good": f"Good work! {pct}% shows solid understanding.",
        "Average": f"You scored {pct}%. Review the weak areas.",
        "Needs Improvement": f"You scored {pct}%. Keep practicing.",
    }
    return {
        "overall_feedback": msgs.get(band, f"You scored {pct}%."),
        "study_recommendations": [f"Review: {q['question'][:70]}" for q in wrong[:3]],
    }


def _empty_result(student_id, test_id, time_taken_seconds) -> dict:
    return {
        "score": 0,
        "total": 0,
        "percentage": 0,
        "time_taken_seconds": time_taken_seconds,
        "performance_band": "Needs Improvement",
        "overall_feedback": "No questions submitted.",
        "results": [],
        "study_recommendations": [],
        "student_id": student_id,
        "test_id": test_id,
    }
