from jose import jwt
import os
from dotenv import load_dotenv

load_dotenv()
SECRET_KEY = os.getenv("JWT_SECRET")

token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwczovL3d3dy5sZXh0b3JhaC1lbGVhcm5pbmcuY29tL2FwL2xhcmF2ZWwvYXBpL2FpdHV0b3ItbG9naW4iLCJpYXQiOjE3NzE2MDEzNjgsImV4cCI6MTc3MTYwNDk2OCwibmJmIjoxNzcxNjAxMzY4LCJqdGkiOiJmVUxNbUtocmQ5QkkwYVZwIiwic3ViIjoiMTE2OCIsInBydiI6IjIzYmQ1Yzg5NDlmNjAwYWRiMzllNzAxYzQwMDg3MmRiN2E1OTc2ZjciLCJpZCI6MTE2OCwic3R1ZGVudF9pZCI6IkxUUi8yMDIyLzA0NTcyNiIsInRpdGxlIjpudWxsLCJzdXJuYW1lIjoiQkFUIiwiZmlyc3RfbmFtZSI6IkJBVCIsIm90aGVyX25hbWUiOm51bGwsImVtYWlsIjoibWVyY3lidGF5bG9yQHlhaG9vLmNvbSIsInBob25lX251bWJlciI6IjA4MDkxMjM0NTY3IiwidXNlcl9sZXZlbCI6IjAiLCJjb3Vyc2UiOiJJRUxUUyIsInJlZ19ieSI6IlNlbGYiLCJjb3VudHJ5IjpudWxsLCJ0b3duIjpudWxsLCJhZGRyZXNzIjpudWxsLCJwYXNzcG9ydCI6bnVsbCwiYWRkcmVzczIiOm51bGwsImhlYXJfdXMiOm51bGx9.g2R6UEczAuGFOFx3K9nlmlCckVkcyZJyFXbs7LCyT_E"

try:
    payload = jwt.decode(
        token, 
        SECRET_KEY, 
        algorithms=["HS256"],
        options={"verify_aud": False, "verify_iss": False}
    )
    print("Valid signature!")
except Exception as e:
    print(f"Error: {e}")
