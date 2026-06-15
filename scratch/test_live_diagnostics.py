import asyncio
import os
from services.aialert import get_course_summary_diagnostics
from utils.protected import UserOutput

async def test_live_diagnostics():
    # Simulate a tutor user
    tutor_user = UserOutput(
        id="tutor_456",
        email="tutor@lextorah.com",
        role="tutor"
    )
    
    # Real course code from database
    course_code = "FRE/A1/WD/177"
    print(f"Calling get_course_summary_diagnostics with course_code: {course_code}...")
    
    try:
        diagnostics = await get_course_summary_diagnostics(course_code=course_code, user=tutor_user)
        print("\nSUCCESS! Diagnostics retrieved:")
        for diag in diagnostics:
            print("-" * 50)
            print(f"ID: {diag.id}")
            print(f"Type: {diag.type}")
            print(f"Message: {diag.message}")
            print(f"Reason: {diag.reason}")
            print("-" * 50)
    except Exception as e:
        print(f"\nERROR running diagnostics: {e}")

if __name__ == "__main__":
    asyncio.run(test_live_diagnostics())
