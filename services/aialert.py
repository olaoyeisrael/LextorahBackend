from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List
from bson import ObjectId
import re
import os
from datetime import datetime
from openai import OpenAI

from utils.protected import decode_token, UserOutput
from utils.mongodb import quiz_results_collection, user_collection

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

router = APIRouter()

class AcademicDeclineAlert(BaseModel):
    id: str  # Format: "courseCode_studentId"
    type: str = "Warning"
    message: str
    reason: str = "Declining Trend"

@router.get("/academic-decline", response_model=List[AcademicDeclineAlert])
async def get_academic_decline_alerts(
    course_code: str = Query(None),
    user: UserOutput = Depends(decode_token)
):
    print(f"User role: {user.role}, User ID: {user.id}, Course Code query: {course_code}")
    if user.role not in ("admin", "tutor"):
        raise HTTPException(status_code=403, detail="Admins and tutors only")
    
    match_filter = {}
    if user.role == "tutor":
        if not course_code:
            return []
            
        requested_codes = [c.strip() for c in course_code.split(",") if c.strip()]
        if not requested_codes:
            return []
            
        regex_list = [f"^{re.escape(c)}$" for c in requested_codes]
        match_filter["course_code"] = {"$in": [re.compile(r, re.IGNORECASE) for r in regex_list]}
        print(f"Match filter for tutor {user.id}: {match_filter}")
        
    elif user.role == "admin":
        if course_code:
            requested_codes = [c.strip() for c in course_code.split(",") if c.strip()]
            regex_list = [f"^{re.escape(c)}$" for c in requested_codes]
            match_filter["course_code"] = {"$in": [re.compile(r, re.IGNORECASE) for r in regex_list]}
            print(f"Match filter for admin: {match_filter}")
       
    
    pipeline = [
        {"$match": match_filter},
        {
            "$project": {
                "student_id": {"$ifNull": ["$student_id", "$user_id"]},
                "course_code": 1,
                "created_at": {"$ifNull": ["$created_at", "$date"]},
                "score_percent": {
                    "$ifNull": [
                        "$score_percent",
                        {
                            "$cond": [
                                {"$gt": ["$total", 0]},
                                {"$multiply": [{"$divide": ["$score", "$total"]}, 100]},
                                0
                            ]
                        }
                    ]
                },
                "student_name": 1
            }
        },
        {"$sort": {"created_at": -1}},
        {
            "$group": {
                "_id": {
                    "student_id": "$student_id",
                    "course_code": "$course_code"
                },
                "scores": {"$push": "$score_percent"},
                "student_names": {"$push": "$student_name"}
            }
        },
        {
            "$project": {
                "scores": {"$slice": ["$scores", 3]},
                "student_name_fallback": {"$arrayElemAt": ["$student_names", 0]}
            }
        },
        {
            "$match": {
                "scores": {"$size": 3}
            }
        },
        {
            "$lookup": {
                "from": "users",
                "let": {"res_student_id": "$_id.student_id"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$eq": [
                                    {"$toString": "$_id"},
                                    "$$res_student_id"
                                ]
                            }
                        }
                    }
                ],
                "as": "student_info"
            }
        },
        {
            "$unwind": {
                "path": "$student_info",
                "preserveNullAndEmptyArrays": True
            }
        }
    ]
    print('course_code filter:', match_filter.get("course_code", "No filter applied"))
    
    cursor = quiz_results_collection.aggregate(pipeline)
    aggregated_results = await cursor.to_list(length=1000)
    print(f"Aggregated results count: {len(aggregated_results)}")
    
    alerts = []
    for doc in aggregated_results:
        scores = doc.get("scores", [])
        if len(scores) < 3:
            continue
            
        # strict decline: most_recent < second_most_recent < oldest
        if scores[0] < scores[1] < scores[2]:
            student_info = doc.get("student_info")
            if student_info:
                first_name = student_info.get("first_name", "")
                last_name = student_info.get("last_name", "")
                student_name = f"{first_name} {last_name}".strip()
                if not student_name:
                    student_name = student_info.get("name", "Student")
            else:
                student_name = doc.get("student_name_fallback") or "Student"
            
            course_code = doc["_id"].get("course_code", "UnknownCourse")
            student_id = doc["_id"].get("student_id", "UnknownStudent")
            
            alerts.append(
                AcademicDeclineAlert(
                    id=f"{course_code}_{student_id}",
                    type="Warning",
                    message=f"{student_name} shows consistent decline over last 3 assessments.",
                    reason="Declining Trend"
                )
            )
            
    return alerts

@router.get("/academic-decline/test", response_model=List[AcademicDeclineAlert])
async def get_academic_decline_alerts_test(
    course_code: str = Query(None),
    user: UserOutput = Depends(decode_token)
):
    print(f"[TEST ENDPOINT] User role: {user.role}, User ID: {user.id}, Course Code query: {course_code}")
    if user.role not in ("admin", "tutor"):
        raise HTTPException(status_code=403, detail="Admins and tutors only")
    
    match_filter = {}
    if user.role == "tutor":
        if not course_code:
            return []
            
        requested_codes = [c.strip() for c in course_code.split(",") if c.strip()]
        if not requested_codes:
            return []
            
        regex_list = [f"^{re.escape(c)}$" for c in requested_codes]
        match_filter["course_code"] = {"$in": [re.compile(r, re.IGNORECASE) for r in regex_list]}
        
    elif user.role == "admin":
        if course_code:
            requested_codes = [c.strip() for c in course_code.split(",") if c.strip()]
            regex_list = [f"^{re.escape(c)}$" for c in requested_codes]
            match_filter["course_code"] = {"$in": [re.compile(r, re.IGNORECASE) for r in regex_list]}
    
    pipeline = [
        {"$match": match_filter},
        {
            "$project": {
                "student_id": {"$ifNull": ["$student_id", "$user_id"]},
                "course_code": 1,
                "created_at": {"$ifNull": ["$created_at", "$date"]},
                "score_percent": {
                    "$ifNull": [
                        "$score_percent",
                        {
                            "$cond": [
                                {"$gt": ["$total", 0]},
                                {"$multiply": [{"$divide": ["$score", "$total"]}, 100]},
                                0
                            ]
                        }
                    ]
                },
                "student_name": 1
            }
        },
        {"$sort": {"created_at": -1}},
        {
            "$group": {
                "_id": {
                    "student_id": "$student_id",
                    "course_code": "$course_code"
                },
                "scores": {"$push": "$score_percent"},
                "student_names": {"$push": "$student_name"}
            }
        },
        {
            "$project": {
                "scores": {"$slice": ["$scores", 1]},
                "student_name_fallback": {"$arrayElemAt": ["$student_names", 0]}
            }
        },
        {
            "$match": {
                "scores": {"$size": 1}
            }
        },
        {
            "$lookup": {
                "from": "users",
                "let": {"res_student_id": "$_id.student_id"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$eq": [
                                    {"$toString": "$_id"},
                                    "$$res_student_id"
                                ]
                            }
                        }
                    }
                ],
                "as": "student_info"
            }
        },
        {
            "$unwind": {
                "path": "$student_info",
                "preserveNullAndEmptyArrays": True
            }
        }
    ]
    
    cursor = quiz_results_collection.aggregate(pipeline)
    aggregated_results = await cursor.to_list(length=1000)
    
    alerts = []
    for doc in aggregated_results:
        student_info = doc.get("student_info")
        if student_info:
            first_name = student_info.get("first_name", "")
            last_name = student_info.get("last_name", "")
            student_name = f"{first_name} {last_name}".strip()
            if not student_name:
                student_name = student_info.get("name", "Student")
        else:
            student_name = doc.get("student_name_fallback") or "Student"
            
        course_code = doc["_id"].get("course_code", "UnknownCourse")
        student_id = doc["_id"].get("student_id", "UnknownStudent")
        
        alerts.append(
            AcademicDeclineAlert(
                id=f"{course_code}_{student_id}",
                type="Warning",
                message=f"{student_name} shows consistent decline over last 3 assessments.",
                reason="Declining Trend"
            )
        )
        
    return alerts

class StudentDiagnosticInput(BaseModel):
    student_id: str = Field(description="The unique identifier of the student")
    student_name: str = Field(description="The student's name")
    risk_level: str = Field(description="Struggling status or risk level (e.g., High, Medium, Low)")
    message: str = Field(description="Actionable and personalized message warning the teacher of the student's learning gaps, performance issues, or declining trends and how to help them.")
    reason: str = Field(description="The primary reason or pattern identified, e.g., 'Declining Trend', 'Grammar struggles', 'Vocabulary comprehension gaps'")

class CourseDiagnosticsResponse(BaseModel):
    common_blindspot: str = Field(description="The primary class-wide common blindspot most students struggle with in this course")
    students: List[StudentDiagnosticInput] = Field(description="List of diagnostics for individual struggling students")

class FlatStudentDiagnostic(BaseModel):
    id: str = Field(description="Unique identifier for the alert, e.g., 'courseCode_studentId'")
    type: str = Field(description="Type of alert, e.g., 'Warning'")
    message: str = Field(description="Actionable message")
    reason: str = Field(description="The underlying reason or pattern identified, e.g., 'Declining Trend'")

@router.get("/api/reports/course-summary/{course_code:path}", response_model=List[FlatStudentDiagnostic])
async def get_course_summary_diagnostics(
    course_code: str,
    user: UserOutput = Depends(decode_token)
):
    print(f"[COURSE DIAGNOSTICS] User role: {user.role}, User ID: {user.id}, Course Code: {course_code}")
    if user.role not in ("admin", "tutor"):
        raise HTTPException(status_code=403, detail="Admins and tutors only")
        
    # Build aggregation pipeline to group assessments by user_id
    pipeline = [
        {"$match": {"course_code": {"$regex": f"^{re.escape(course_code.strip())}$", "$options": "i"}}},
        {
            "$project": {
                "student_id": {"$ifNull": ["$student_id", "$user_id"]},
                "score_percent": {
                    "$ifNull": [
                        "$score_percent",
                        {
                            "$cond": [
                                {"$gt": ["$total", 0]},
                                {"$multiply": [{"$divide": ["$score", "$total"]}, 100]},
                                0
                            ]
                        }
                    ]
                },
                "student_name": 1,
                "topic": 1,
                "date": {"$ifNull": ["$date", "$created_at"]}
            }
        },
        {"$sort": {"date": -1}},
        {
            "$group": {
                "_id": "$student_id",
                "student_name": {"$first": "$student_name"},
                "history": {
                    "$push": {
                        "topic": "$topic",
                        "score_percent": "$score_percent",
                        "date": "$date"
                    }
                }
            }
        }
    ]
    
    cursor = quiz_results_collection.aggregate(pipeline)
    student_histories = await cursor.to_list(length=1000)
    print('StudentHistories:', student_histories)
    
    if not student_histories:
        return []
        
    # Compile text representation of histories for LLM
    prompt = f"Analyze the quiz and exam performance history of students in course '{course_code}' to identify class-wide common blindspots and run diagnostics on struggling students.\n\n"
    prompt += "STUDENTS PERFORMANCE DATA:\n"
    for student in student_histories:
        name = student.get("student_name") or "Student"
        sid = student.get("_id") or "UnknownID"
        prompt += f"- Student: {name} (ID: {sid})\n"
        prompt += "  History:\n"
        for entry in student.get("history", []):
            topic = entry.get("topic") or "General Assessment"
            score = entry.get("score_percent", 0.0)
            date_val = entry.get("date")
            date_str = date_val.isoformat() if isinstance(date_val, datetime) else str(date_val)
            prompt += f"    * Date: {date_str}, Topic: {topic}, Score: {score:.1f}%\n"
    print('Prompt:', prompt)       
    # Call OpenAI Structured Outputs
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful education diagnostics assistant. Analyze the students' academic histories and output structured diagnostics for the class."},
                {"role": "user", "content": prompt}
            ],
            response_format=CourseDiagnosticsResponse
        )
        parsed_response = completion.choices[0].message.parsed
    except Exception as e:
        print(f"Failed to generate AI course summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate AI diagnostics: {str(e)}")
        
    if not parsed_response:
        raise HTTPException(status_code=500, detail="Empty response from AI diagnostics model")
        
    # Flatten the response structure
    flat_alerts = []
    for s in parsed_response.students:
        # We only want to alert on students who actually need help (High/Medium risk)
        if s.risk_level.lower() in ["high", "medium"]:
            flat_alerts.append(
                FlatStudentDiagnostic(
                    id=f"{course_code.replace('/', '_')}_{s.student_id}",
                    type="Warning", 
                    message=s.message,
                    reason=s.reason
                )
            )
        
    return flat_alerts
