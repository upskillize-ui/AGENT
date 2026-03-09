/**
 * mockTestService.js — Node.js / Express
 * Connects your React frontend ↔ RAG FastAPI agent.
 * No content bundling needed — agent reads from its own vector store.
 */

const axios = require("axios");

const AGENT = process.env.MOCK_TEST_AGENT_URL || "http://localhost:8000";
const TIMEOUT_GENERATE = 60_000; // 60s — RAG + Claude generation
const TIMEOUT_SUBMIT = 30_000;

// ─── Ingest triggers ──────────────────────────────────────────────────────────
// Call these from your existing lecture/note/PDF upload handlers.

/**
 * Call after a lecture is created OR its content is updated.
 * The agent ingests in the background — your upload endpoint returns immediately.
 */
async function triggerLectureIngest(lectureId) {
  try {
    await axios.post(`${AGENT}/api/ingest/lecture`, { lecture_id: lectureId });
    console.log(`[MockTest] Ingestion triggered for lecture ${lectureId}`);
  } catch (err) {
    console.error(`[MockTest] Ingest trigger failed:`, err.message);
    // Non-fatal — content will be missing from vector store until retried
  }
}

/**
 * Call when an entire course is published or bulk-uploaded.
 */
async function triggerCourseIngest(courseId) {
  try {
    await axios.post(`${AGENT}/api/ingest/course`, { course_id: courseId });
    console.log(`[MockTest] Course ingestion triggered for course ${courseId}`);
  } catch (err) {
    console.error(`[MockTest] Course ingest trigger failed:`, err.message);
  }
}

// ─── Route handlers ───────────────────────────────────────────────────────────

/**
 * POST /api/mock-test/generate
 * Student picks topic + config → agent retrieves relevant chunks → Claude generates test
 */
async function generateMockTest(req, res) {
  const {
    lectureId,
    courseId,
    topic,
    numQuestions = 10,
    durationMinutes = 30,
    difficulty = "medium",
    questionTypes = ["mcq"],
  } = req.body;

  if (!lectureId && !courseId) {
    return res.status(400).json({ error: "Provide lectureId or courseId" });
  }
  if (!topic) {
    return res.status(400).json({ error: "topic is required" });
  }

  try {
    const { data } = await axios.post(
      `${AGENT}/api/generate-test`,
      {
        topic,
        lecture_id: lectureId || null,
        course_id: courseId || null,
        num_questions: numQuestions,
        duration_minutes: durationMinutes,
        difficulty,
        question_types: questionTypes,
        student_id: String(req.user.id),
      },
      { timeout: TIMEOUT_GENERATE },
    );
    return res.json(data);
  } catch (err) {
    const status = err.response?.status || 500;
    const message = err.response?.data?.detail || err.message;
    console.error("[MockTest] generate error:", message);
    return res.status(status).json({ error: message });
  }
}

/**
 * POST /api/mock-test/submit
 * Student submits answers → agent evaluates → returns score + feedback
 */
async function submitMockTest(req, res) {
  const { testId, questions, answers, timeTakenSeconds } = req.body;

  try {
    const { data } = await axios.post(
      `${AGENT}/api/submit-answers`,
      {
        test_id: testId,
        student_id: String(req.user.id),
        questions,
        answers,
        time_taken_seconds: timeTakenSeconds,
      },
      { timeout: TIMEOUT_SUBMIT },
    );

    // Optional: save to your MySQL results table
    // await db.query(
    //   "INSERT INTO test_results (student_id, test_id, score, percentage, created_at) VALUES (?,?,?,?,NOW())",
    //   [req.user.id, testId, data.score, data.percentage]
    // );

    return res.json(data);
  } catch (err) {
    console.error("[MockTest] submit error:", err.message);
    return res.status(500).json({ error: err.message });
  }
}

module.exports = {
  generateMockTest,
  submitMockTest,
  triggerLectureIngest,
  triggerCourseIngest,
};

// ─── Wire up routes ────────────────────────────────────────────────────────────
// In your Express app.js:
//
// const mockTest = require("./services/mockTestService");
// app.post("/api/mock-test/generate", authMiddleware, mockTest.generateMockTest);
// app.post("/api/mock-test/submit",   authMiddleware, mockTest.submitMockTest);
//
// ─── Wire up ingest triggers ──────────────────────────────────────────────────
// In your lecture upload handler:
//
// const { triggerLectureIngest } = require("./services/mockTestService");
//
// router.post("/lectures", authMiddleware, async (req, res) => {
//   const lecture = await Lecture.create(req.body);          // your existing code
//   await triggerLectureIngest(lecture.id);                  // ← ADD THIS
//   res.json(lecture);
// });
