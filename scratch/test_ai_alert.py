import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from services.aialert import get_academic_decline_alerts, get_academic_decline_alerts_test, get_course_summary_diagnostics, AcademicDeclineAlert
from utils.protected import UserOutput

class TestAcademicDeclineAlerts(unittest.IsolatedAsyncioTestCase):

    async def test_academic_decline_alerts_tutor_success(self):
        mock_aggregated_results = [
            # Student 1: Consistent Decline trend (50 < 70 < 90) -> Alert expected!
            {
                "_id": {"student_id": "student_1", "course_code": "GER/A1/WD/899"},
                "scores": [50.0, 70.0, 90.0],
                "student_name_fallback": "John Fallback",
                "student_info": {
                    "first_name": "John",
                    "last_name": "Doe"
                }
            },
            # Student 2: Non-decline trend (80 >= 70 < 75) -> No alert!
            {
                "_id": {"student_id": "student_2", "course_code": "GER/A1/WD/899"},
                "scores": [80.0, 70.0, 75.0],
                "student_name_fallback": "Jane Fallback",
                "student_info": {
                    "first_name": "Jane",
                    "last_name": "Smith"
                }
            },
            # Student 3: Steady trend (80, 80, 80) -> No alert!
            {
                "_id": {"student_id": "student_3", "course_code": "GER/A1/WD/899"},
                "scores": [80.0, 80.0, 80.0],
                "student_name_fallback": "Alice Johnson",
                "student_info": None
            }
        ]
        
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=mock_aggregated_results)
        
        with patch("services.aialert.quiz_results_collection") as mock_quiz_col:
            mock_quiz_col.aggregate = MagicMock(return_value=mock_cursor)
            
            current_user = UserOutput(
                id="tutor_123",
                email="tutor@test.com",
                role="tutor"
            )
            
            alerts = await get_academic_decline_alerts(course_code="GER/A1/WD/899,FRE/B1/WD/987", user=current_user)
            
            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0].id, "GER/A1/WD/899_student_1")
            self.assertEqual(alerts[0].type, "Warning")
            self.assertEqual(alerts[0].message, "John Doe shows consistent decline over last 3 assessments.")
            self.assertEqual(alerts[0].reason, "Declining Trend")
            
            # Verify matching stage was filtered by frontend supplied courses
            args, kwargs = mock_quiz_col.aggregate.call_args
            pipeline = args[0]
            match_stage = pipeline[0]["$match"]
            regex_list = match_stage["course_code"]["$in"]
            self.assertEqual(len(regex_list), 2)
            self.assertEqual(regex_list[0].pattern, "^GER/A1/WD/899$")
            self.assertEqual(regex_list[1].pattern, "^FRE/B1/WD/987$")

    async def test_academic_decline_alerts_tutor_empty_query(self):
        current_user = UserOutput(
            id="tutor_123",
            email="tutor@test.com",
            role="tutor"
        )
        
        alerts = await get_academic_decline_alerts(course_code="", user=current_user)
        self.assertEqual(alerts, [])
        
        alerts_none = await get_academic_decline_alerts(course_code=None, user=current_user)
        self.assertEqual(alerts_none, [])

    async def test_academic_decline_alerts_unauthorized(self):
        current_user = UserOutput(
            id="student_1",
            email="student@test.com",
            role="student"
        )
        
        with self.assertRaises(HTTPException) as context:
            await get_academic_decline_alerts(course_code="GER/A1/WD/899", user=current_user)
        
        self.assertEqual(context.exception.status_code, 403)
        self.assertEqual(context.exception.detail, "Admins and tutors only")

    async def test_academic_decline_alerts_test_endpoint(self):
        mock_aggregated_results = [
            # Student 1: Test variant checks size 1 -> triggers alert directly
            {
                "_id": {"student_id": "student_1", "course_code": "GER/A1/WD/899"},
                "scores": [50.0],
                "student_name_fallback": "Alice Fallback",
                "student_info": None
            }
        ]
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=mock_aggregated_results)
        
        with patch("services.aialert.quiz_results_collection") as mock_quiz_col:
            mock_quiz_col.aggregate = MagicMock(return_value=mock_cursor)
            
            current_user = UserOutput(
                id="tutor_123",
                email="tutor@test.com",
                role="tutor"
            )
            
            alerts = await get_academic_decline_alerts_test(course_code="GER/A1/WD/899", user=current_user)
            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0].message, "Alice Fallback shows consistent decline over last 3 assessments.")

    async def test_get_course_summary_diagnostics_success(self):
        mock_aggregated_history = [
            {
                "_id": "student_1",
                "student_name": "Alice Johnson",
                "history": [
                    {"topic": "Greetings", "score_percent": 85.0, "date": "2026-06-12"},
                    {"topic": "Verbs", "score_percent": 50.0, "date": "2026-06-10"}
                ]
            }
        ]
        
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=mock_aggregated_history)
        
        # Mock OpenAI chat completion parse output
        mock_student_diagnostic = MagicMock()
        mock_student_diagnostic.student_id = "student_1"
        mock_student_diagnostic.student_name = "Alice Johnson"
        mock_student_diagnostic.risk_level = "High"
        mock_student_diagnostic.message = "Alice Johnson is identified as High risk. Core issue: Struggling with conjugating helper verbs."
        mock_student_diagnostic.reason = "Conjugation gaps"
        
        mock_response = MagicMock()
        mock_response.common_blindspot = "Understanding negative sentences"
        mock_response.students = [mock_student_diagnostic]
        
        mock_choice = MagicMock()
        mock_choice.message.parsed = mock_response
        
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        
        with patch("services.aialert.quiz_results_collection") as mock_quiz_col, \
             patch("services.aialert.client.beta.chat.completions.parse") as mock_parse:
            
            mock_quiz_col.aggregate = MagicMock(return_value=mock_cursor)
            mock_parse.return_value = mock_completion
            
            current_user = UserOutput(
                id="tutor_123",
                email="tutor@test.com",
                role="tutor"
            )
            
            diagnostics = await get_course_summary_diagnostics(course_code="GER/A1/WD/899", user=current_user)
            
            self.assertEqual(len(diagnostics), 1)
            self.assertEqual(diagnostics[0].id, "GER_A1_WD_899_student_1")
            self.assertEqual(diagnostics[0].type, "Warning")
            self.assertEqual(diagnostics[0].message, "Alice Johnson is identified as High risk. Core issue: Struggling with conjugating helper verbs.")
            self.assertEqual(diagnostics[0].reason, "Conjugation gaps")

if __name__ == "__main__":
    unittest.main()
