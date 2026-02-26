from jose import jwt, JWTError
from pydantic import BaseModel

from fastapi import HTTPException, status, Header, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from dotenv import load_dotenv
import os
load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET")


class UserOutput(BaseModel):
    id: str
    email: str
    role: str | None = None
    first_name: str | None = None
    last_name: str | None = None 
    enrolled_course: str | None = None
    enrolled_level: str | None = None


def get_token_from_header(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication")
    return authorization.split("Bearer ")[1]

def decode_token(token: str = Depends(get_token_from_header)):
    if token and token.startswith("Bearer "):
        token = token.split("Bearer ")[1]
    try:
        payload = jwt.decode(
            token, 
            SECRET_KEY, 
            algorithms=["HS256"],
            options={"verify_aud": False, "verify_iss": False, "verify_nbf": False}
        )
        student_id = payload.get("student_id")
        if student_id and str(student_id).lower() == "admin":
            role = "admin"
        else:
            role = "student"
            
        print("Payload: ",payload)
        print("Role: ",role)
        print("Email: ",payload.get("email"))
        print("ID: ",payload.get("sub"))
      
        return UserOutput(
            id=str(payload.get("id")),
            email=payload.get("email"),
            role=role,
            first_name=payload.get("first_name"),
            last_name=payload.get("surname"),
            enrolled_course=payload.get("course"),
            enrolled_level=payload.get("level", "A1")
        )
    except JWTError as e:
        with open("jwt_debug_log.txt", "a") as f:
            f.write(f"JWTError triggered for token: {token}\n")
            f.write(f"Detail: {str(e)}\n")
        print(f"JWTError triggered for token: {token}")
        print(f"Detail: {str(e)}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication")