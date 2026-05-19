from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from typing import Optional
from datetime import datetime
import os
from pydantic import BaseModel
from bson import ObjectId
from utils.mongodb import assignments_collection, assignment_submissions_collection
from services.cloudinary_client import uploadMaterialToCloudinary

router = APIRouter()

@router.post("/assignments")
async def create_assignment(
    topic: str = Form(...),
    course_code: str = Form(...),
    question_text: str = Form(...),
    tutor_id: str = Form(...),
    deadline: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    try:
        cloud_url = None
        if file:
            os.makedirs("materials", exist_ok=True)
            filepath = os.path.join("materials", file.filename)
            with open(filepath, "wb") as f:
                content = await file.read()
                f.write(content)
            cloud_url = uploadMaterialToCloudinary(filepath)

        assignment_doc = {
            "topic": topic,
            "course_code": course_code,
            "question_text": question_text,
            "tutor_id": tutor_id,
            "deadline": deadline,
            "file_url": cloud_url,
            "created_at": datetime.now()
        }

        inserted = await assignments_collection.insert_one(assignment_doc)
        return {"message": "Assignment created successfully", "id": str(inserted.inserted_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/assignments")
async def get_assignments(course_code: Optional[str] = None, tutor_id: Optional[str] = None):
    try:
        query = {}
        if course_code:
            codes = [c.strip() for c in course_code.split(",")]
            query["course_code"] = {"$in": codes}
            print("Filtering by course codes:", codes)
        if tutor_id:
            query["tutor_id"] = tutor_id
            print("Filtering by tutor ID:", tutor_id)
        
        cursor = assignments_collection.find(query).sort("created_at", -1)
        assignments = await cursor.to_list(length=100)
        
        # Convert ObjectId to string for JSON serialization
        for assignment in assignments:
            assignment["_id"] = str(assignment["_id"])
            
        return {"assignments": assignments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/assignments/{assignment_id}/submit")
async def submit_assignment(
    assignment_id: str,
    student_id: str = Form(...),
    student_name: str = Form("Unknown Student"),
    answer_text: str = Form(""),
    file: Optional[UploadFile] = File(None)
):
    try:
        # Check for duplicate submission
        existing = await assignment_submissions_collection.find_one({
            "assignment_id": assignment_id,
            "student_id": student_id
        })
        if existing:
            raise HTTPException(status_code=409, detail="You have already submitted this assignment.")
            
        # Check deadline
        assignment = await assignments_collection.find_one({"_id": ObjectId(assignment_id)})
        if assignment and assignment.get("deadline"):
            try:
                deadline_date = datetime.fromisoformat(assignment["deadline"])
                if datetime.now() > deadline_date:
                    raise HTTPException(status_code=403, detail="The deadline for this assignment has passed.")
            except ValueError:
                pass # Ignore parsing errors for old or invalid dates

        cloud_url = None
        if file:
            os.makedirs("materials", exist_ok=True)
            filepath = os.path.join("materials", file.filename)
            with open(filepath, "wb") as f:
                content = await file.read()
                f.write(content)
            cloud_url = uploadMaterialToCloudinary(filepath)

        submission_doc = {
            "assignment_id": assignment_id,
            "student_id": student_id,
            "student_name": student_name,
            "answer_text": answer_text,
            "file_url": cloud_url,
            "submitted_at": datetime.now()
        }

        inserted = await assignment_submissions_collection.insert_one(submission_doc)
        return {"message": "Assignment submitted successfully", "id": str(inserted.inserted_id)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/assignments/{assignment_id}/submissions")
async def get_submissions(assignment_id: str):
    try:
        cursor = assignment_submissions_collection.find({"assignment_id": assignment_id}).sort("submitted_at", -1)
        submissions = await cursor.to_list(length=100)
        
        for sub in submissions:
            sub["_id"] = str(sub["_id"])
            
        return {"submissions": submissions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
