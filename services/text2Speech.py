from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel

class Input(BaseModel):
    input: str

client = OpenAI(api_key=os.getenv('OPEN_AI_KEY'))
speech_file_path = Path(__file__).parent / "../speech/speech.mp3"
def TTS(input: str):
   
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="shimmer",
        input=input,
        instructions="Speak in a cheerful and positive tone.",
    ) as response:
        response.stream_to_file(speech_file_path)

# fable


# ash