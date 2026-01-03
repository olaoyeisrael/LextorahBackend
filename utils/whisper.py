import whisper

def transcribe_audio(filepath):
    model = whisper.load_model("base")
    result = model.transcribe(filepath)
    return result['text']