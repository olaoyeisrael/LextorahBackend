from fastapi import FastAPI, Query, UploadFile, File
from pydantic import BaseModel
from services.model import ask_with_history
from services.uploadMaterial import uploadMaterial
import os 
from PyPDF2 import PdfReader


class TeachRequest(BaseModel):
    filename: str


from typing import Union
import io

def extract_text_from_file(file_input: Union[str, io.BytesIO]):
    reader = PdfReader(file_input)
    sections = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            sections.append({"title": f"Section {i+1}", "content": text})
    return sections






def teach(filename: str):
    filepath = os.path.join("materials", filename)
    if not os.path.exists(filepath):
        return {"error": "File not found."}
    sections = extract_text_from_file(filepath)
    teaching_plan = []
    for section in sections:
#         prompt = (
#     f"Give a brief, simple summary and explanation of this section. "
#     f"No filler, no extra remarks.\n\n{section['content']}"
# )
        prompt = (
    "Teach this section to a student in short, simple, direct sentences. "
    "Explain it the way a teacher talks to a student. "
    "Do NOT summarize. "
    "Do NOT use headings, bullets, or numbered lists. "
    "Do NOT restate key points. "
    'Do NOT include words like "overview", "summary", "in conclusion", or any closing remarks. '
    "Just explain the meaning in a short, beginner-friendly way.\n\n"
    "Example of the style:\n"
    "Student: What does this section mean?\n"
    "Teacher: It is basically saying that the system uses sensors to understand how the patient is positioned. "
    "It watches changes over time so it can notice risks early.\n\n"
    "Now explain this section in the same style:\n\n"
    f"{section['content']}"
)




        explanation = ask_with_history("1-chat", prompt)
        teaching_plan.append({
            "section": section["title"],
            "explanation": explanation
        })
    return teaching_plan
    