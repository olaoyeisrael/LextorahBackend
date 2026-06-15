import os
from jose import jwt
from dotenv import load_dotenv

# Load env variables
load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET")
if not SECRET_KEY:
    SECRET_KEY = "6XbVIV9SYkySX4WOiqOvk4U5ooALSMCMlzOKX021IOdDsQsU6FYduA0CR7RD0nm0"

# Admin payload
# decode_token looks for:
# payload.get("student_id") to decide role
# payload.get("id") -> user id
# payload.get("email") -> email
# payload.get("first_name") -> first name
# payload.get("surname") -> last name
admin_payload = {
    "student_id": "admin",
    "id": "admin_123",
    "email": "admin@lextorah.com",
    "first_name": "System",
    "surname": "Admin",
}

# Tutor payload (starts with tutor to assign tutor role)
tutor_payload = {
    "student_id": "tutor_456",
    "id": "tutor_456",
    "email": "tutor@lextorah.com",
    "first_name": "Jane",
    "surname": "Tutor",
}

admin_token = jwt.encode(admin_payload, SECRET_KEY, algorithm="HS256")
tutor_token = jwt.encode(tutor_payload, SECRET_KEY, algorithm="HS256")

print("="*60)
print("ADMIN TOKEN:")
print("="*60)
print(admin_token)
print("\n" + "="*60)
print("TUTOR TOKEN:")
print("="*60)
print(tutor_token)
print("="*60)
