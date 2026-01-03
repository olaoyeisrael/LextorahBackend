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
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        if payload.get("student_id").lower() == "admin" :
            role = "admin"
        else:
            role = "student"
            
        print("Payload: ",payload)
        print("Role: ",role)
        print("Email: ",payload.get("email"))
        print("ID: ",payload.get("sub"))
      
        return UserOutput(
            id=payload.get("sub"),
            email=payload.get("email"),
            role=role,
            first_name=payload.get("first_name"),
            last_name=payload.get("surname"),
            enrolled_course=payload.get("course"),
            enrolled_level=payload.get("level", "A1")
        )
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication")