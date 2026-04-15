from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List
from datetime import datetime
import os
import json
import random
from pydantic import BaseModel
from bson import ObjectId
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
from utils.mongodb import exam_questions_collection, exam_results_collection
from services.cloudinary_client import uploadMaterialToCloudinary
import io

load_dotenv()
client = OpenAI(api_key=os.getenv('OPEN_AI_KEY'))

router = APIRouter()

# ─── AI Prompt for parsing exam material into structured questions ───
PARSE_QUESTIONS_PROMPT = """You are an expert exam question parser. Analyze the following exam material and extract ALL individual questions from it.

For each question, classify it as one of:
- "objective" — multiple choice question with 4 options
- "fill_in_gap" — a sentence with a blank that needs to be filled in

Output ONLY a valid JSON array. Each object must have these fields:
- "type": "objective" or "fill_in_gap"
- "question": the question text (for fill_in_gap, include the blank as "___")
- "options": array of 4 option strings (for objective only, use empty array [] for fill_in_gap)
- "answer": the correct answer
- "explanation": brief explanation of why this is correct

IMPORTANT: Extract EVERY question you can find. For fill-in-the-gap questions where a verb or word needs to be conjugated/transformed, the answer should be the correctly conjugated/transformed form.

MATERIAL:
{content}
"""

# ─── AI Prompt for grading answers ───
GRADE_ANSWERS_PROMPT = """You are an expert exam grader. Grade the following student answers against the correct answers.

For each question, determine if the student's answer is correct. Be lenient with:
- Minor spelling differences
- Capitalization differences
- For fill-in-the-gap: accept valid alternative forms if semantically correct

Return ONLY a valid JSON array where each object has:
- "question_id": the question id provided
- "correct": true or false
- "student_answer": what the student answered
- "correct_answer": the actual correct answer
- "feedback": brief feedback explaining why it's correct or wrong

QUESTIONS AND ANSWERS:
{qa_data}
"""


def extract_text_from_pdf(file_content):
    """Extract text from PDF bytes."""
    pdf_reader = PdfReader(io.BytesIO(file_content))
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def parse_questions_with_ai(content: str) -> list:
    """Use OpenAI to parse exam material into structured questions."""
    prompt = PARSE_QUESTIONS_PROMPT.format(content=content)
    
    res = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )
    
    try:
        # Try to extract JSON from the response
        output = res.output_text.strip()
        # Handle markdown code blocks
        if output.startswith("```"):
            output = output.split("\n", 1)[1]
            output = output.rsplit("```", 1)[0]
        questions = json.loads(output)
        return questions if isinstance(questions, list) else []
    except Exception as e:
        print(f"Failed to parse AI response: {e}")
        print(f"Raw response: {res.output_text}")
        return []


def grade_answers_with_ai(qa_data: list) -> list:
    """Use OpenAI to grade student answers."""
    prompt = GRADE_ANSWERS_PROMPT.format(qa_data=json.dumps(qa_data, indent=2))
    
    res = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )
    
    try:
        output = res.output_text.strip()
        if output.startswith("```"):
            output = output.split("\n", 1)[1]
            output = output.rsplit("```", 1)[0]
        results = json.loads(output)
        return results if isinstance(results, list) else []
    except Exception as e:
        print(f"Failed to parse grading response: {e}")
        return []


# ─── ENDPOINTS ───

@router.post("/exam-questions/upload")
async def upload_exam_questions(
    course_code: str = Form(...),
    tutor_id: str = Form(...),
    question_text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """Tutor uploads exam material. AI parses it into structured questions."""
    try:
        content = ""
        cloud_url = None

        if file:
            file_content = await file.read()
            # Upload to cloudinary for reference
            os.makedirs("materials", exist_ok=True)
            filepath = os.path.join("materials", file.filename)
            with open(filepath, "wb") as f:
                f.write(file_content)
            cloud_url = uploadMaterialToCloudinary(filepath)
            
            # Extract text from PDF
            if file.filename.lower().endswith(".pdf"):
                content = extract_text_from_pdf(file_content)
            else:
                content = file_content.decode("utf-8", errors="ignore")

        if question_text:
            content = question_text if not content else content + "\n\n" + question_text

        if not content.strip():
            raise HTTPException(status_code=400, detail="No content provided. Upload a file or enter question text.")

        # Parse questions with AI
        parsed_questions = parse_questions_with_ai(content)

        if not parsed_questions:
            raise HTTPException(status_code=422, detail="AI could not parse any questions from the provided material.")

        # Store each question individually in the pool
        inserted_count = 0
        for q in parsed_questions:
            doc = {
                "course_code": course_code,
                "tutor_id": tutor_id,
                "type": q.get("type", "objective"),
                "question": q.get("question", ""),
                "options": q.get("options", []),
                "answer": q.get("answer", ""),
                "explanation": q.get("explanation", ""),
                "source_file_url": cloud_url,
                "created_at": datetime.now()
            }
            await exam_questions_collection.insert_one(doc)
            inserted_count += 1

        return {
            "message": f"Successfully parsed and stored {inserted_count} questions.",
            "count": inserted_count,
            "questions": parsed_questions
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exam-questions")
async def get_exam_questions(course_code: Optional[str] = None):
    """Fetch all questions from the pool (tutor overview)."""
    try:
        query = {}
        if course_code:
            codes = [c.strip() for c in course_code.split(",")]
            query["course_code"] = {"$in": codes}
        
        cursor = exam_questions_collection.find(query).sort("created_at", -1)
        questions = await cursor.to_list(length=500)
        
        for q in questions:
            q["_id"] = str(q["_id"])
        
        return {"questions": questions, "total": len(questions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exam-questions/practice")
async def get_practice_questions(course_code: str):
    """Get 10 random questions for practice mode."""
    try:
        codes = [c.strip() for c in course_code.split(",")]
        pipeline = [
            {"$match": {"course_code": {"$in": codes}}},
            {"$sample": {"size": 10}}
        ]
        cursor = exam_questions_collection.aggregate(pipeline)
        questions = await cursor.to_list(length=10)
        
        for q in questions:
            q["_id"] = str(q["_id"])
        
        return {"questions": questions, "mode": "practice", "total": len(questions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exam-questions/mock")
async def get_mock_questions(course_code: str):
    """Get 15 random questions for timed mock exam."""
    try:
        codes = [c.strip() for c in course_code.split(",")]
        pipeline = [
            {"$match": {"course_code": {"$in": codes}}},
            {"$sample": {"size": 15}}
        ]
        cursor = exam_questions_collection.aggregate(pipeline)
        questions = await cursor.to_list(length=15)
        
        for q in questions:
            q["_id"] = str(q["_id"])
        
        return {"questions": questions, "mode": "mock", "total": len(questions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AnswerItem(BaseModel):
    question_id: str
    question: str
    correct_answer: str
    student_answer: str
    type: str = "objective"

class ExamResultRequest(BaseModel):
    student_id: str
    course_code: str
    mode: str  # "practice" or "mock"
    answers: List[AnswerItem]

@router.post("/exam-results")
async def submit_exam_results(body: ExamResultRequest):
    """AI grades the student's answers and saves the result."""
    try:
        # Prepare data for AI grading
        qa_data = []
        for ans in body.answers:
            qa_data.append({
                "question_id": ans.question_id,
                "question": ans.question,
                "correct_answer": ans.correct_answer,
                "student_answer": ans.student_answer,
                "type": ans.type
            })

        # Grade with AI
        grading_results = grade_answers_with_ai(qa_data)

        # Calculate score
        correct_count = sum(1 for r in grading_results if r.get("correct", False))
        total = len(body.answers)
        score_percent = round((correct_count / total) * 100, 1) if total > 0 else 0

        # Save to DB
        result_doc = {
            "student_id": body.student_id,
            "course_code": body.course_code,
            "mode": body.mode,
            "score": correct_count,
            "total": total,
            "score_percent": score_percent,
            "details": grading_results,
            "submitted_at": datetime.now()
        }

        inserted = await exam_results_collection.insert_one(result_doc)

        return {
            "message": "Exam graded successfully",
            "id": str(inserted.inserted_id),
            "score": correct_count,
            "total": total,
            "score_percent": score_percent,
            "details": grading_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
