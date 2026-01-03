
from openai import OpenAI
import json
from dotenv import load_dotenv
import os
load_dotenv()

QUIZ_PROMPT_MATERIAL = """
You are an expert tutor. Create a quiz of {num_questions} high-quality questions based on the material below.

REQUIREMENTS:
- Question types: multiple choice.
- Questions should test understanding, not just memorization.
- Difficulty: {difficulty}
- Each question must include a clear answer.
- Output in JSON array format:
[
  {{
    "question": "...",
    "options": ["A ...", "B ...", "C ...", "D ..."], 
    "answer": "...",
    "explanation": "..."
    }}
]

MATERIAL:
{content}
"""

QUIZ_PROMPT_TOPIC = """
You are an expert tutor. Create a quiz of {num_questions} high-quality questions on the topic: "{topic}" (Subject: {subject}).

REQUIREMENTS:
- Question types: multiple choice.
- Questions should test understanding, not just memorization.
- Difficulty: {difficulty}
- Each question must include a clear answer.
- Each question must include an explanation for why the answer is correct.
- Output in JSON array format:
[
  {{
    "question": "...",
    "options": ["A ...", "B ...", "C ...", "D ..."], 
    "answer": "...",
    "explanation": "..."
    }}
]
"""
client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

import asyncio

def generate_quiz(material: str = None, topic: str = None, subject: str = "General", difficulty: str = "Medium", num_questions: int = 5):
    if material:
        print("Material: ", material)
        prompt = QUIZ_PROMPT_MATERIAL.format(content=material, difficulty=difficulty, num_questions=num_questions)
    elif topic:
        prompt = QUIZ_PROMPT_TOPIC.format(topic=topic, subject=subject, difficulty=difficulty, num_questions=num_questions)
    else:
        return []

    res =  client.responses.create(
        model= "gpt-5-nano",
        input= prompt
    )

    # Optional: validate JSON or fallback
    print(res)
    try:
        quiz = json.loads(res.output_text)
        return quiz
    except:
        # If model returns text instead of JSON, return raw or try to parse
        return res.output_text