import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException
from bson import ObjectId

from services.assignments import grade_submission, GradeSubmissionRequest, create_assignment
from utils.protected import UserOutput

class TestAssignmentGrading(unittest.IsolatedAsyncioTestCase):

    async def test_grade_submission_success_tutor(self):
        # Mock submission and assignment data
        mock_submission = {
            "_id": ObjectId("660000000000000000000001"),
            "assignment_id": "660000000000000000000002",
            "student_id": "student_123",
            "student_name": "John Doe",
            "answer_text": "Here is my answer",
            "file_url": None
        }
        
        mock_assignment = {
            "_id": ObjectId("660000000000000000000002"),
            "tutor_id": "tutor_123",
            "max_points": 50
        }
        
        with patch("services.assignments.assignment_submissions_collection") as mock_sub_col, \
             patch("services.assignments.assignments_collection") as mock_assign_col:
            
            mock_sub_col.find_one = AsyncMock(return_value=mock_submission)
            mock_assign_col.find_one = AsyncMock(return_value=mock_assignment)
            mock_sub_col.update_one = AsyncMock()
            
            tutor_user = UserOutput(
                id="tutor_123",
                email="tutor@test.com",
                role="tutor"
            )
            
            payload = GradeSubmissionRequest(
                score=45.0,
                feedback="Well done!",
                status="graded"
            )
            
            res = await grade_submission(
                submission_id="660000000000000000000001",
                payload=payload,
                user=tutor_user
            )
            
            self.assertEqual(res["message"], "Submission graded successfully")
            self.assertEqual(res["score"], 45.0)
            self.assertEqual(res["feedback"], "Well done!")
            mock_sub_col.update_one.assert_called_once()

    async def test_grade_submission_success_admin(self):
        mock_submission = {
            "_id": ObjectId("660000000000000000000001"),
            "assignment_id": "660000000000000000000002",
            "student_id": "student_123"
        }
        
        # Assignment belongs to tutor_123, but grader is admin
        mock_assignment = {
            "_id": ObjectId("660000000000000000000002"),
            "tutor_id": "tutor_123",
            "max_points": 100
        }
        
        with patch("services.assignments.assignment_submissions_collection") as mock_sub_col, \
             patch("services.assignments.assignments_collection") as mock_assign_col:
            
            mock_sub_col.find_one = AsyncMock(return_value=mock_submission)
            mock_assign_col.find_one = AsyncMock(return_value=mock_assignment)
            mock_sub_col.update_one = AsyncMock()
            
            admin_user = UserOutput(
                id="admin_999",
                email="admin@test.com",
                role="admin"
            )
            
            payload = GradeSubmissionRequest(score=95.0, feedback="Excellent work by admin!")
            
            res = await grade_submission(
                submission_id="660000000000000000000001",
                payload=payload,
                user=admin_user
            )
            
            self.assertEqual(res["message"], "Submission graded successfully")
            self.assertEqual(res["score"], 95.0)
            mock_sub_col.update_one.assert_called_once()

    async def test_grade_submission_unauthorized_student(self):
        mock_submission = {
            "_id": ObjectId("660000000000000000000001"),
            "assignment_id": "660000000000000000000002",
            "student_id": "student_123"
        }
        
        mock_assignment = {
            "_id": ObjectId("660000000000000000000002"),
            "tutor_id": "tutor_123"
        }
        
        with patch("services.assignments.assignment_submissions_collection") as mock_sub_col, \
             patch("services.assignments.assignments_collection") as mock_assign_col:
            
            mock_sub_col.find_one = AsyncMock(return_value=mock_submission)
            mock_assign_col.find_one = AsyncMock(return_value=mock_assignment)
            
            student_user = UserOutput(
                id="student_123",
                email="student@test.com",
                role="student"
            )
            
            payload = GradeSubmissionRequest(score=80.0)
            
            with self.assertRaises(HTTPException) as ctx:
                await grade_submission(
                    submission_id="660000000000000000000001",
                    payload=payload,
                    user=student_user
                )
            
            self.assertEqual(ctx.exception.status_code, 403)
            self.assertIn("Only tutors and admins are authorized", ctx.exception.detail)

    async def test_grade_submission_unauthorized_tutor_mismatch(self):
        mock_submission = {
            "_id": ObjectId("660000000000000000000001"),
            "assignment_id": "660000000000000000000002",
            "student_id": "student_123"
        }
        
        # Assignment belongs to tutor_123, but tutor_456 is trying to grade
        mock_assignment = {
            "_id": ObjectId("660000000000000000000002"),
            "tutor_id": "tutor_123"
        }
        
        with patch("services.assignments.assignment_submissions_collection") as mock_sub_col, \
             patch("services.assignments.assignments_collection") as mock_assign_col:
            
            mock_sub_col.find_one = AsyncMock(return_value=mock_submission)
            mock_assign_col.find_one = AsyncMock(return_value=mock_assignment)
            
            tutor_user = UserOutput(
                id="tutor_456",
                email="tutor456@test.com",
                role="tutor"
            )
            
            payload = GradeSubmissionRequest(score=80.0)
            
            with self.assertRaises(HTTPException) as ctx:
                await grade_submission(
                    submission_id="660000000000000000000001",
                    payload=payload,
                    user=tutor_user
                )
            
            self.assertEqual(ctx.exception.status_code, 403)
            self.assertIn("You do not have permission to grade this specific assignment", ctx.exception.detail)

    async def test_grade_submission_exceeds_max_points(self):
        mock_submission = {
            "_id": ObjectId("660000000000000000000001"),
            "assignment_id": "660000000000000000000002",
            "student_id": "student_123"
        }
        
        mock_assignment = {
            "_id": ObjectId("660000000000000000000002"),
            "tutor_id": "tutor_123",
            "max_points": 20
        }
        
        with patch("services.assignments.assignment_submissions_collection") as mock_sub_col, \
             patch("services.assignments.assignments_collection") as mock_assign_col:
            
            mock_sub_col.find_one = AsyncMock(return_value=mock_submission)
            mock_assign_col.find_one = AsyncMock(return_value=mock_assignment)
            
            tutor_user = UserOutput(
                id="tutor_123",
                email="tutor@test.com",
                role="tutor"
            )
            
            payload = GradeSubmissionRequest(score=25.0, max_points=20.0) # 25 exceeds max of 20
            
            with self.assertRaises(HTTPException) as ctx:
                await grade_submission(
                    submission_id="660000000000000000000001",
                    payload=payload,
                    user=tutor_user
                )
            
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertIn("exceeds the maximum allowed points", ctx.exception.detail)

    async def test_grade_submission_negative_score(self):
        mock_submission = {
            "_id": ObjectId("660000000000000000000001"),
            "assignment_id": "660000000000000000000002",
            "student_id": "student_123"
        }
        
        mock_assignment = {
            "_id": ObjectId("660000000000000000000002"),
            "tutor_id": "tutor_123",
            "max_points": 100
        }
        
        with patch("services.assignments.assignment_submissions_collection") as mock_sub_col, \
             patch("services.assignments.assignments_collection") as mock_assign_col:
            
            mock_sub_col.find_one = AsyncMock(return_value=mock_submission)
            mock_assign_col.find_one = AsyncMock(return_value=mock_assignment)
            
            tutor_user = UserOutput(
                id="tutor_123",
                email="tutor@test.com",
                role="tutor"
            )
            
            payload = GradeSubmissionRequest(score=-5.0)
            
            with self.assertRaises(HTTPException) as ctx:
                await grade_submission(
                    submission_id="660000000000000000000001",
                    payload=payload,
                    user=tutor_user
                )
            
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertIn("Score cannot be negative", ctx.exception.detail)

    async def test_create_assignment_without_max_points(self):
        with patch("services.assignments.assignments_collection") as mock_assign_col:
            mock_insert_result = MagicMock()
            mock_insert_result.inserted_id = ObjectId("660000000000000000000003")
            mock_assign_col.insert_one = AsyncMock(return_value=mock_insert_result)
            
            res = await create_assignment(
                topic="French Verb Conjugation",
                course_code="FRE-A1",
                question_text="Conjugate the verb 'être' in present tense.",
                tutor_id="tutor_123",
                deadline="2026-06-30T23:59:59",
                file=None
            )
            
            self.assertEqual(res["message"], "Assignment created successfully")
            self.assertEqual(res["id"], "660000000000000000000003")
            
            mock_assign_col.insert_one.assert_called_once()
            inserted_doc = mock_assign_col.insert_one.call_args[0][0]
            self.assertEqual(inserted_doc["topic"], "French Verb Conjugation")
            self.assertNotIn("max_points", inserted_doc)

if __name__ == "__main__":
    unittest.main()
