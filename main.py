from fastapi import FastAPI, Query, UploadFile, File, WebSocket, WebSocketDisconnect, Form, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from services.model import ask_with_history
from services.uploadMaterial import uploadMaterial
from services.teaching import teach, TeachRequest, extract_text_from_file
from services.chunker import chunk_text
from services.text2Speech import TTS, Input
from utils.helper import get_user
import os
from fastapi.middleware.cors import CORSMiddleware
from services.chat import chat_with_model
# from services.teaching import stream_teach
# from services.teaching import stream_teach
from fastapi.responses import StreamingResponse

import asyncio
# from services.teach import generate_lesson_stream
from utils.mongodb import user_collection
from services.quiz import generate_quiz
# from services.transcription import transcribe_audio
import shutil
from utils.protected import decode_token
from services.pdf_generator import create_pdf_from_text
from datetime import datetime
import io
from services.pdf_generator import create_pdf_from_text
from datetime import datetime
import io
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv('OPEN_AI_KEY'))


# from services.cloudinary import uploadMaterial



app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/teach/{userid}")
async def teach_ws(websocket: WebSocket, userid: str):
    await websocket.accept()
    try:
        # ---- INITIAL MESSAGE ----
        init = await websocket.receive_json()
        print(f"DEBUG: Received init payload: {init}")

        filename = init.get("filename")
        topic = init.get("topic")
        user_id = userid
        auto = bool(init.get("auto", False))

        # ---- FETCH USER ----
        user = await get_user(user_id)
        
        # Determine file from topic if provided
        sections = []

        if not filename and topic and user:
            from utils.mongodb import course_collection
            # Find material for this topic specific to user's course/level?
            # Ideally material is tied to course/level/topic.
            # Assuming user enrollment matches material metadata.
            curr_course = user.get("enrolled_course", "German")
            curr_level = user.get("enrolled_level", "A1")
            
            print(f"DEBUG: Looking for material - Topic: {topic}, Course: {curr_course}, Level: {curr_level}")
            
            mat = await course_collection.find_one({
                "topic": {"$regex": f"^{topic}$", "$options": "i"},
                "course_title": {"$regex": f"^{curr_course}$", "$options": "i"},
                "level": {"$regex": f"^{curr_level}$", "$options": "i"}
            })
            
            if mat:
                if mat.get("cloud_url"):
                    import requests
                    import io
                    c_url = mat.get("cloud_url")
                    print(f"DEBUG: Using Cloudinary URL: {c_url}")
                    
                    filename = c_url.split('/')[-1]
                    if '?' in filename: filename = filename.split('?')[0]
                    
                    def download_to_memory(url):
                         try:
                             r = requests.get(url)
                             if r.status_code == 200:
                                 return io.BytesIO(r.content)
                         except Exception as e:
                             print(f"Download Error: {e}")
                         return None
                    
                    pdf_stream = await asyncio.to_thread(download_to_memory, c_url)
                    
                    if not pdf_stream:
                         await websocket.send_json({"error": "failed to download material"})
                         await websocket.close()
                         return
                         
                    sections = extract_text_from_file(pdf_stream)
                    
                elif mat.get("file"):
                    filename = os.path.basename(mat.get("file"))
            
            if not filename:
                 await websocket.send_json({"error": f"Material for topic '{topic}' not found."})
                 await websocket.close()
                 return   

        # Check if filename provided directly is a URL
        elif filename and filename.startswith("http"):
            print(f"DEBUG: Filename provided is URL: {filename}")
            import requests
            import io
            
            c_url = filename
            # Update filename to basename for tracking
            filename = c_url.split('/')[-1]
            if '?' in filename: filename = filename.split('?')[0]
            
            def download_to_memory(url):
                    try:
                        r = requests.get(url)
                        if r.status_code == 200:
                            return io.BytesIO(r.content)
                    except Exception as e:
                        print(f"Download Error: {e}")
                    return None
            
            pdf_stream = await asyncio.to_thread(download_to_memory, c_url)
            
            if not pdf_stream:
                    await websocket.send_json({"error": "failed to download material from provided URL"})
                    await websocket.close()
                    return
            
            sections = extract_text_from_file(pdf_stream)   

        print(user)
        # ---- FIX: CORRECT start_section LOGIC ----
        if user is None:
            # Brand new user → always start from 1 or provided start_section
            idx = int(init.get("start_section", 1))
        
        else:
            # # Existing user → use provided start_section if included
            # if "start_section" in init:
            #     idx = int(init["start_section"])
            # else:
            idx = int(user.get("current_section", 1))

        if filename:
            safe = os.path.normpath(filename)

        # ---- VALIDATE FILE (If not already loaded) ----
        if not sections:
            if not filename:
                await websocket.send_json({"error": "missing filename"})
                await websocket.close()
                return

            # safe is already set above
            if os.path.isabs(safe) or safe.startswith(".."):
                await websocket.send_json({"error": "invalid filename"})
                await websocket.close()
                return

            path = os.path.join("materials", safe)
            if not os.path.exists(path):
                await websocket.send_json({"error": "file not found"})
                await websocket.close()
                return
            
            sections = extract_text_from_file(path)

        # ---- UPDATE USER STARTING POSITION ----
        await user_collection.update_one(
            {"_id": user_id},
            {"$set": {"file": filename, "current_section": idx}},
            upsert=True
        )
        total = len(sections)

        if total == 0:
            await websocket.send_json({"error": "no readable sections"})
            await websocket.close()
            return
        
        # ---- SEND SYLLABUS ----
        await websocket.send_json({
            "type": "syllabus",
            "sections": [{"index": i + 1, "title": s.get("title", f"Section {i+1}")} for i, s in enumerate(sections)]
        })

        # ---- ACCUMULATE EXPLANATIONS ----
        generated_explanations = []

        # ---- MAIN LOOP ----
        while 1 <= idx <= total:

            # save current user progress
            await user_collection.update_one(
                {"_id": user_id},
                {"$set": {"current_section": idx, "file": filename}}
            )

            section = sections[idx - 1]

            # ---- GENERATE EXPLANATION (non-blocking) ----
            prompt = (
                f"Teach this section to a student clearly and simply:\n\n"
                f"{section['content']}"
            )

            explanation = await asyncio.to_thread(
                ask_with_history,
                f"teach-{os.path.basename(safe)}-{idx}",
                prompt
            )

            generated_explanations.append(explanation)

            # ---- SEND EXPLANATION TO CLIENT ----
            await websocket.send_json({
                "type": "explanation",
                "section_index": idx,
                "explanation": explanation
            })

            # ---- AUTO MODE ----
            if auto:
                idx += 1
                continue

            # ---- WAIT FOR CLIENT COMMAND ----
            msg = await websocket.receive_json()
            cmd = msg.get("cmd")

            if cmd == "next":
                idx += 1

            elif cmd == "repeat":
                generated_explanations.pop() # Remove the last one if repeating, to avoid dups?
                if generated_explanations:
                     generated_explanations.pop()
                continue

            elif cmd == "chat":
                question = msg.get("message")
                answer = await chat_with_model(
                    f"{user_id}",
                    question
                )
                await websocket.send_json({
                    "type": "chat_answer",
                    "message": answer
                })

            elif cmd == "goto":
                idx = int(msg.get("section", idx))

            elif cmd == "close":
                await websocket.close()
                return

            else:
                # command not recognized → do nothing
                continue
        
        # ---- LESSON COMPLETE: GENERATE QUIZ ----
        # Use generated explanations if available, else fallback
        if generated_explanations:
             full_content = "\n\n".join(generated_explanations)
             print("Using generated explanations for quiz.")
        else:
             full_content = "\n\n".join([s["content"] for s in sections])
             print("Using raw content for quiz (no explanations generated).")

        print("full content length: ", len(full_content))
        
        await websocket.send_json({
            "type": "info", 
            "message": "Generating quiz..."
        })

        quiz_data = await asyncio.to_thread(generate_quiz, full_content)
        # quiz_data is already the parsed JSON/list from the service


        print("Quiz Data:", quiz_data)

        await websocket.send_json({
            "type": "quiz",
            "quiz": quiz_data
        })
        
        # Keep connection open for quiz interaction if needed, or close?
        # The frontend might want to close or just show the quiz.
        # Let's wait for a close command or just hang out.
        while True:
             msg = await websocket.receive_json()
             if msg.get("cmd") == "close":
                 break


        # ---- SAVE TRANSCRIPT ----
        if generated_explanations:
             full_content = "\n\n".join(generated_explanations)
             transcript_entry = {
                 "user_id": user_id,
                 "file": filename,
                 "topic": safe, # Use safe filename as topic or extract title
                 "content": full_content,
                 "date": datetime.now()
             }
             # Could store in a separate collection or within user document
             # For now, let's store in a 'transcripts' collection
             from utils.mongodb import client
             db = client["lextorah"] # adjust db name if needed, assuming user_collection uses same db
             await db.transcripts.insert_one(transcript_entry)
             print(f"Transcript saved for {user_id}")


    except WebSocketDisconnect:
        return

    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
            await websocket.close()
        except:
            pass

@app.get('/')
async def home():
    return "Welcome to Lextorah Ai"

class QuizRequest(BaseModel):
    material : str = None
    topic: str = None
    subject: str = "General"
    difficulty: str = "Medium"
    num_questions: int = 5


@app.post('/quiz')
async def quiz_endpoint(body: QuizRequest):
    # Offload to thread if blocking, though generate_quiz seems synchronous in current implementation 
    # (actually it uses client.responses.create which might be sync or async depending on lib version, 
    # but let's wrap it to be safe if it's sync)
    material_content = body.material
    if material_content and material_content.startswith("http") and material_content.endswith(".pdf"):
        print(f"Downloading from provided Cloud URL: {material_content}")
        import requests
        # We need a temporary filename or hash
        filename = material_content.split('/')[-1]
        safe = os.path.normpath(filename)
        path = os.path.join("materials", safe)
        
        # Determine if we need to download
        need_download = True
        if os.path.exists(path):
            need_download = False
            # Optional: check if valid pdf?

        if need_download:
            def download_file(url, dest):
                try:
                    r = requests.get(url)
                    if r.status_code == 200:
                        os.makedirs("materials", exist_ok=True)
                        with open(dest, "wb") as f:
                            f.write(r.content)
                        return True
                except: return False
                return False
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, download_file, material_content, path)
        
        if os.path.exists(path):
             # Extract text for quiz context
             material_content = extract_text_from_file(path)
    elif material_content and material_content.endswith(".pdf"):
        # Assume local if not http (though mostly will be http if from cloud)
        if os.path.exists(material_content):
            material_content = extract_text_from_file(material_content)

    
    # Check if at least material or topic is provided
    if not material_content and not body.topic:
         return {"error": "Please provide either material or a topic."}

    res = await asyncio.to_thread(
        generate_quiz, 
        material=material_content, 
        topic=body.topic, 
        subject=body.subject, 
        difficulty=body.difficulty,
        num_questions=body.num_questions
    )
    
    return {
        "Quiz": res
    }

    



class QuestionRequest(BaseModel):
    question: str

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Save temp file
    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        audio_file = open(temp_filename, "rb")
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format="text"
        )
        audio_file.close()
        return {"text": transcription}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


# @app.post('/ask')
# async def askQuestion(chatid: str = Query(..., alias="chatid"), body: QuestionRequest = None):
#     res = await chat_with_model(chatid, body.question)   
#     return  {"answer": res}

class UserOutput(BaseModel):
    id: str  # Changed from int
    email: str
    role: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    enrolled_course: str | None = None
    enrolled_level: str | None = None
    current_topic_index: int = 0
    progress: dict = {} # { "German-A1": { "completed_topics": [], "current_topic": "..." } }

    current_section: int | None = 1
    file: str | None = None # Current file being learned


@app.delete('/chat_history')
async def clear_chat_history(user: UserOutput = Depends(decode_token)):
    from utils.mongodb import user_collection
    
    await user_collection.update_one(
        {"_id": user.id},
        {"$set": {"history_chat": []}}
    )
    return {"msg": "Chat history cleared"}

@app.post('/ask')
async def askQuestion(chatid: str = Query(..., alias="chatid"), body: QuestionRequest = None, user: UserOutput = Depends(decode_token)):
    print(user.id)
    if user.role == "admin":
        return {"error": "Admins cannot ask questions"}
    if not body or not body.question:
        raise HTTPException(status_code=400, detail="Missing question")
    
    # ✅ Await async call
    # Use authenticated user.id instead of potentially invalid chatid
    res = await chat_with_model(user.id, body.question)

    
   
    return {"answer": res}




@app.post('/upload')

async def upload_material_endpoint(
    course_title: str = Form(...), 
    topic: str = Form(...),
    skill: str = Form(...),
    level: str = Form(...),
    file: UploadFile = File(...),
    user: UserOutput = Depends(decode_token)
):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admins only")

    res = await uploadMaterial(course_title.lower(), topic, skill, level.lower(), file)

    return {"msg": res}

@app.post('/teach')
async def teach_endpoint(request: TeachRequest):
    result = teach(request.filename)
    return {"teaching_plan": result}

@app.post('/getAudio')
async def get_audio(input: Input):
    print(input)
    TTS(input.input)
    return FileResponse('speech/speech.mp3', media_type="audio/mp3")

# @app.get('/teach/stream')
# async def teach_stream_endpoint(filename: str = Query(..., alias="filename"), start_section: int = Query(1, alias="start_section")):
#     """
#     Stream teaching sections as SSE. Client should connect and listen for `text/event-stream`.
#     Example: GET /teach/stream?filename=course.pdf&start_section=1
#     """
#     generator = stream_teach(filename, start_section)
#     return StreamingResponse(generator, media_type="text/event-stream")



@app.get('/history/{user_id}')
async def get_activity_history(user_id: str):
    from utils.mongodb import history_collection
    cursor = history_collection.find({"user_id": user_id}).sort("date", -1)
    history = await cursor.to_list(length=100)
    
    for h in history:
        h["id"] = str(h["_id"])
        del h["_id"]
        
    return {"History" : history}

@app.get('/chat_history/{user_id}')
async def get_chat_history(user_id: str):
    from utils.mongodb import user_collection
    
    user = await user_collection.find_one({"_id": user_id})
    if not user:
        return {"History": []}
        
    history = user.get("history_chat", [])
    if history and "_id" in history[0]:
         for h in history:
             h["id"] = str(h["_id"])
             del h["_id"]
             
    return {"History" : history}


@app.post('/decode')
async def decode_token_endpoint(token: str = Query(..., alias="token")):
    decodedToken = decode_token(token)
    
    # Sync User to DB
    update_data = {
        "email": decodedToken.email,
        "first_name": decodedToken.first_name,
        "last_name": decodedToken.last_name,
        "role": decodedToken.role
    }
    
    if decodedToken.enrolled_course:
        update_data["enrolled_course"] = decodedToken.enrolled_course
        
    if decodedToken.enrolled_level:
         update_data["enrolled_level"] = decodedToken.enrolled_level

    result = await user_collection.update_one(
        {"_id": decodedToken.id},
        {"$set": update_data},
        upsert=True
    )
    
    return {"decoded": decodedToken}


class SupportChatRequest(BaseModel):
    session_id: str
    question: str

@app.post('/support')
async def support_chat_endpoint(body: SupportChatRequest):
    from services.chat import chat_support
    res = await chat_support(body.session_id, body.question)
    return {"answer": res}


@app.post("/chat_voice")
async def chat_voice(file: UploadFile = File(...), user: UserOutput = Depends(decode_token)):
    # Save temp file
    temp_filename = f"temp_{user.id}_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Transcribe
    text = transcribe_audio(temp_filename)
    
    # Clean up
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
        
    if not text:
        raise HTTPException(status_code=500, detail="Transcription failed")

   
    
    answer = await chat_with_model(user.id, text)
    
    # Optionally generate audio for the answer
    # For now just return text
    return {"text": text, "answer": answer}



# --- COURSE SCHEDULE & ENROLLMENT ---

# Hardcoded simplified schedule data for prototype
# German A1-B2, 4 weeks, 3 topics/week (approx)
GERMAN_SCHEDULE = {
    "A1": {
        "Week 1": ["Basics & Greetings", "Numbers & Dates", "Personal Pronouns"],
        "Week 2": ["Present Tense Verbs", "Articles (Der/Die/Das)", "Sentence Structure"],
        "Week 3": ["Accusative Case", "Modal Verbs", "Prepositions of Place"],
        "Week 4": ["Family & Home", "Food & Drink", "Daily Routine"]
    },
    "A2": {
        "Week 1": ["Past Tense (Perfekt)", "Dative Case", "Reflexive Verbs"],
        "Week 2": ["Adjective Endings", "Comparative & Superlative", "Future Tense"],
        "Week 3": ["Subordinate Clauses", "Relative Clauses", "Conjunctions"],
        "Week 4": ["Travel & Holidays", "Health & Body", "Shopping & Clothing"]
    },
    "B1": {
        "Week 1": ["Passive Voice", "Genitive Case", "Past Perfect (Plusquamperfekt)"],
        "Week 2": ["Infinitive with zu", "Participle Clauses", "N-Declension"],
        "Week 3": ["Complex Sentence Structures", "Idiomatic Expressions", "Conditional (Konjunktiv II)"],
        "Week 4": ["Work & Career", "Media & Technology", "Environment & Nature"]
    },
    "B2": {
        "Week 1": ["Advanced Passive Forms", "Nominalization", "Subjunctive I (Konjunktiv I)"],
        "Week 2": ["Fixed Prepositions", "Extended Attribute", "Gerunds"],
        "Week 3": ["Connectors & Particles", "Style & Register", "Debating & Argumentation"],
        "Week 4": ["Politics & Society", "Art & Culture", "Science & History"]
    }
    }


class CurriculumItem(BaseModel):
    course: str
    level: str
    week: int | None = None
    topic: str
    description: str | None = None
    index: int | None = None # Optional, auto-assigned if missing


@app.post('/curriculum')
async def add_curriculum(item: CurriculumItem, user: UserOutput = Depends(decode_token)):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admins only")
        
    from utils.mongodb import curriculum_collection
    
    # Check if exists
    # If index is manually provided, check conflict? 
    # For now just insert.
    entry = item.dict()
    entry["course"] = entry["course"].lower()
    entry["level"] = entry["level"].lower()
    
    # Auto-assign index if missing?
    if item.index is None:
         # simple count
         count = await curriculum_collection.count_documents({"course": item.course.lower(), "level": item.level.lower()})
         entry["index"] = count
         
    entry["created_at"] = datetime.now()
    
    await curriculum_collection.insert_one(entry)
    return {"msg": "Added curriculum item"}

@app.post('/curriculum/batch')
async def add_curriculum_batch(items: list[CurriculumItem], user: UserOutput = Depends(decode_token)):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admins only")
        
    from utils.mongodb import curriculum_collection
    
    entries = []
    # We need to handle indexing.
    # It's complex with distinct course/levels in one batch?
    # Assume batch is consistent or handle per item.
    
    # Let's count current items to determine start index?
    # Or just let them be inserted and sorted by week/topic naturally?
    # Our `get_curriculum_list` sorts by `index`.
    # If index is missing, we should auto-assign.
    
    for item in items:
        entry = item.dict()
        if item.index is None:
             # This is slow in loop but safer for order
             count = await curriculum_collection.count_documents({"course": item.course, "level": item.level})
             entry["index"] = count
             
        entry["created_at"] = datetime.now()
        entries.append(entry)
        
        # Artificial small delay or just rely on sequential processing to not race condition too much?
        # Actually count_documents might race if parallel.
        # But we are in one async request. DB ops are awaited.
        # If we insert immediately after count, it's safeish.
        print("Inserting: ", entry)
        await curriculum_collection.insert_one(entry)
    
    return {"msg": f"{len(items)} items added"}

@app.get('/curriculum')

async def get_curriculum_list(course: str = Query(None), level: str = Query(None)):
    from utils.mongodb import curriculum_collection
    
    query = {}
    if course: query["course"] = {"$regex": f"^{course}$", "$options": "i"}
    if level: query["level"] = {"$regex": f"^{level}$", "$options": "i"}
    
    cursor = curriculum_collection.find(query).sort("index", 1)

    items = await cursor.to_list(length=1000)
    
    for i in items:
        i["id"] = str(i["_id"])
        del i["_id"]
        
    return {"curriculum": items}


class EnrollmentRequest(BaseModel):
    course: str # e.g., "German"
    level: str  # e.g., "A1"

@app.post('/enroll')
async def enroll_user(req: EnrollmentRequest, user: UserOutput = Depends(decode_token)):
    # Update user in DB with enrollment info
    await user_collection.update_one(
        {"_id": user.id},
        {"$set": {"enrolled_course": req.course.lower(), "enrolled_level": req.level.lower()}}
    )
    return {"msg": f"Enrolled in {req.course.lower()} {req.level.lower()}"}

@app.get('/schedule')
async def get_schedule(user: UserOutput = Depends(decode_token)):
    # Fetch user to get enrolled level
    db_user = await get_user(user.id)
    level = db_user.get("enrolled_level")
    course = db_user.get("enrolled_course")
    

    if not level or not course:
        return {"enrolled": False}
    
    # Return schedule for that level
    # For now assuming course is "German"
    if course and course.lower() == "german" and level in GERMAN_SCHEDULE:
         return {"enrolled": True, "level": level, "course": course, "schedule": GERMAN_SCHEDULE[level]}
    
    return {"enrolled": True, "level": level, "course": course, "schedule": {}}


    
    return {"enrolled": True, "level": level, "course": course, "schedule": {}}


@app.get('/progress')
async def get_progress(user: UserOutput = Depends(decode_token)):
    # Calculate progress based on current_section / total_sections of the current file
    # This is a bit tricky because 'file' in user model is the *current* file.
    # If we want course progress, we need to know how many files/topics are in the course.
    # For now, let's implement the specific request: "My courses should be the course and the progress in percentage... with the section the user is in and the total sections"
    
    # We will use the 'file' the user is currently on (or last worked on) to determine the "active" topic progress.
    # For overall course progress, we'd need a syllabus.
    
    # Let's return the progress of the *current active topic* for now, effectively. 
    # Or if we have the full schedule, we could calculate overall.
    
    if not user.enrolled_level or not user.enrolled_course:
         return {"progress": 0, "current_section": 0, "total_sections": 0, "course": "None"}

    current_file = user.file
    current_section = user.current_section or 1
    
    if not current_file:
         return {"progress": 0, "current_section": 0, "total_sections": 0, "course": f"{user.enrolled_course} {user.enrolled_level}"}

    # Get total sections for the file
    path = os.path.join("materials", os.path.normpath(current_file))
    if os.path.exists(path):
        sections = extract_text_from_file(path)
        total = len(sections)
    else:
        total = 1 # avoid div by zero

    progress = int((current_section / total) * 100) if total > 0 else 0
    
    return {
        "course": f"{user.enrolled_course} {user.enrolled_level}",
        "current_topic": os.path.basename(current_file), # simple name
        "progress": progress,
        "current_section": current_section,
        "total_sections": total
    }

@app.get('/transcripts')
async def get_transcripts(user: UserOutput = Depends(decode_token)):
    from utils.mongodb import db
    # db is lextorahmaterials
    cursor = db.transcripts.find({"user_id": user.id})
    transcripts = await cursor.to_list(length=100)

    print("Transcripts: ", transcripts)
    
    # Convert ObjectId to str
    for t in transcripts:
        t["id"] = str(t["_id"])
        del t["_id"]
        
    return {"transcripts": transcripts}

@app.get('/download_transcript/{transcript_id}')
async def download_transcript(transcript_id: str):
    from utils.mongodb import db
    from bson import ObjectId

    
    try:
        t = await db.transcripts.find_one({"_id": ObjectId(transcript_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid ID")

    if not t:
        raise HTTPException(status_code=404, detail="Transcript not found")
        
    pdf_bytes = create_pdf_from_text(t["content"], f"{t['topic']}.pdf")
    
    return StreamingResponse(
        io.BytesIO(pdf_bytes), 
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=transcript_{t['topic']}.pdf"}
    )

from fastapi import Body
class QuizResultDB(BaseModel):
    user_id: str | None = None
    topic: str | None = None # Optional, inferred or provided
    level: str
    course_title: str
    score: int
    total: int
    date: datetime | None = None
    details: list[dict] | None = None

@app.post('/submit_quiz_result')
async def submit_quiz_result(result: QuizResultDB, user: UserOutput = Depends(decode_token)):
    from utils.mongodb import quiz_results_collection
    
    data = result.dict()
    data["user_id"] = user.id
    data["student_name"] = f"{user.first_name} {user.last_name}"
    data["date"] = datetime.now()
    
    await quiz_results_collection.insert_one(data)
    return {"msg": "Quiz result saved"}

class TopicCompletionRequest(BaseModel):
    topic: str
    course: str
    level: str
    material_content: str | None = None

class LiveClass(BaseModel):
    course: str
    level: str
    week: str
    topic: str
    date: str # YYYY-MM-DD
    time: str # HH:MM AM/PM
    duration: int # minutes
    meeting_link: str
    recording_link: str | None = None
    tutor: str | None = None 

@app.post("/live_class")
async def schedule_live_class(cls: LiveClass, user: UserOutput = Depends(decode_token)):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admins only")
    
    from utils.mongodb import live_classes_collection
    
    # Simple conversion to dict
    data = cls.dict()
    data["course"] = data["course"].lower()
    data["level"] = data["level"].lower()
    data["created_at"] = datetime.now()
    
    # Insert
    res = await live_classes_collection.insert_one(data)
    return {"msg": "Class scheduled", "id": str(res.inserted_id)}

@app.get("/live_classes")
async def get_live_classes(course: str = None, level: str = None):
    from utils.mongodb import live_classes_collection
    
    query = {}
    if course: query["course"] = {"$regex": f"^{course}$", "$options": "i"}
    if level: query["level"] = {"$regex": f"^{level}$", "$options": "i"}
    print("Live Class Update Data: ", live_classes_collection)
    
    cursor = live_classes_collection.find(query).sort("date", 1)
    classes = await cursor.to_list(length=100)
    print("Live Classes: ", classes)
    
    for c in classes:
        c["id"] = str(c["_id"])
        del c["_id"]
        
    return {"live_classes": classes}

@app.put("/live_class/{class_id}")
async def update_live_class(class_id: str, update_data: dict, user: UserOutput = Depends(decode_token)):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admins only")
        
    from utils.mongodb import live_classes_collection
    from bson import ObjectId
    
    # Filter out empty fields if necessary, or just update what's passed
    # For now, just allow updating functionality
    
    await live_classes_collection.update_one(
        {"_id": ObjectId(class_id)},
        {"$set": update_data}
    )
    return {"msg": "Class updated"}

@app.post('/complete_live_class')
async def complete_live_class(body: dict, user: UserOutput = Depends(decode_token)):
    from utils.mongodb import history_collection
    
    entry = {
         "user_id": user.id,
         "type": "live_class", 
         "topic": body.get("topic"),
         "date": datetime.now(),
         "details": body
    }
    await history_collection.insert_one(entry)
    return {"msg": "Live class marked as complete"}

@app.post('/complete_topic_legacy')
async def complete_topic(req: TopicCompletionRequest, user: UserOutput = Depends(decode_token)):
    from utils.mongodb import user_course_collection, user_collection
    
    # 1. Update User Course Progress in User Collection (Topic Tracking)
    course_key = f"{req.course}-{req.level}"
    
    await user_collection.update_one(
        {"_id": user.id},
        {"$addToSet": {f"progress.{course_key}.completed_topics": req.topic}}
    )
    
    # 2. Check complete status
    total_topics = 0
    # Use global schedule if available, else standard count or assume incomplete?
    # For German course:
    if req.course == "German" and req.level in GERMAN_SCHEDULE:
        weeks = GERMAN_SCHEDULE[req.level]
        for w in weeks.values():
            total_topics += len(w)
    
    # Get user's completed topics to compare
    u = await user_collection.find_one({"_id": user.id})
    completed_topics_list = u.get("progress", {}).get(course_key, {}).get("completed_topics", [])
    
    # If total_topics is 0 (unknown course/schedule), we might fallback to just boolean from request, 
    # but for now let's rely on schedule logic.
    is_course_complete = (len(completed_topics_list) >= total_topics) and (total_topics > 0)
    
    # 3. Update/Insert UserCourseDB (Schema: user_id, level, course_title, completed)
    await user_course_collection.update_one(
        {"user_id": user.id, "level": req.level, "course_title": req.course},
        {
            "$set": {
                "completed": is_course_complete, 
                "course_title": req.course, 
                "level": req.level,
                "user_id": user.id,
                "last_updated": datetime.now()
            }
        },
        upsert=True
    )
    
    return {"msg": "Topic completed", "course_completed": is_course_complete}

@app.get('/student_performance')
async def get_student_performance(user: UserOutput = Depends(decode_token)):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admins only")
        
    from utils.mongodb import quiz_results_collection, user_course_collection
    
    # Fetch Quiz Results
    cursor = quiz_results_collection.find().sort("date", -1)
    results = await cursor.to_list(length=200)
    
    # Map _id to id
    for r in results:
        r["id"] = str(r["_id"])
        del r["_id"]
        # Format date for frontend convenience if needed, or send ISO string
        if isinstance(r.get("date"), datetime):
            r["date"] = r["date"].isoformat()

    # We could also fetch course completion statuses if needed, 
    # but the current requirement focuses on "result of each students based on their assessments" (quizzes).
    # The user also asked for "User Course DB" separately. 
    # If admin needs to see course completion, we can add that too.
    # For now, let's return the quiz performance as primary data.
    
    if results and len(results) > 0:
        print("DEBUG: Sample Result Details:", results[0].get("details"))

    return {"performance": results}
    
    # Let's mock or fetch from a 'quiz_results' collection
    cursor = db.quiz_results.find({})
    results = await cursor.to_list(length=100)
    
    # Enrich with user details if needed, or assume quiz_results has name
    for r in results:
        r["id"] = str(r["_id"])
        del r["_id"]
        
    return {"performance": results}

@app.post('/save_quiz_result')
async def save_quiz_result(result: dict, user: UserOutput = Depends(decode_token)):
    # { "score": 5, "total": 10, "course": "German", "topic": "Basics" }
    from utils.mongodb import db
    
    entry = {

        "user_id": user.id,
        "student_name": f"{user.first_name} {user.last_name}",
        "score": result.get("score"),
        "total": result.get("total"),
        "course": result.get("course"),
        "topic": result.get("topic"),
        "date": datetime.now()
    }
    return {"msg": "Saved"}


@app.post('/complete_topic')
async def complete_topic_endpoint(
    body: dict, 
    user: UserOutput = Depends(decode_token)
):
    """
    Marks a topic as complete for the user.
    Updates progress and ensures transcript availability.
    """
    topic = body.get("topic")
    course = body.get("course")
    level = body.get("level")
    material_content = body.get("material_content", "") # For transcript generation if needed
    
    if not topic or not course or not level:
        raise HTTPException(status_code=400, detail="Missing topic, course, or level")

    course_key = f"{course}-{level}"
   
    user_progress =  {}
    print("User progress (raw):", user_progress)
    
    if course_key not in user_progress:
        user_progress[course_key] = {"completed_topics": [], "current_topic": topic}
    
    # Mark complete
    if topic not in user_progress[course_key]["completed_topics"]:
        user_progress[course_key]["completed_topics"].append(topic)
        
    # Calculate next topic
    # We need the full schedule to know what's next
    schedule = await get_flattened_schedule_async(course, level)
    next_topic = None
    curr_idx = -1
    
    for i, item in enumerate(schedule):
        if item["topic"] == topic:
            curr_idx = i
            break
            
    if curr_idx != -1 and curr_idx + 1 < len(schedule):
        next_topic = schedule[curr_idx + 1]["topic"]
    
    # Update current topic pointer
    if next_topic:
        user_progress[course_key]["current_topic"] = next_topic
        # Also update legacy/root level field for backward compatibility
        # We need to calculate global index or just increment if consistent
        # user.current_topic_index = curr_idx + 1
    
    # Save Transcript (if material provided or simply acknowledge completion)
    # The requirement: "save the material to the db to be generated as transcript"
    if material_content:
        # Check if transcript already exists to avoid duplicates
        from utils.mongodb import db
        existing_transcript = await db.transcripts.find_one({
            "user_id": user.id, 
            "topic": topic,
            "course": course,
            "level": level
        })
        
        if not existing_transcript:
            await db.transcripts.insert_one({
                "user_id": user.id,
                "topic": topic,
                "course": course,
                "level": level,
                "content": material_content,
                "date": datetime.now()
            })

    # DB Update
    # DB Update
    await user_collection.update_one(
        {"_id": user.id}, 
        {
            "$set": {
                "progress": user_progress, 
                "current_topic_index": curr_idx + 1,
                "current_section": 1 # Reset session to beginning
            }
        }
    )
    
    # HISTORY & COMPLETION CHECK
    from utils.mongodb import history_collection
    
    # Add Topic Completion History
    await history_collection.insert_one({
        "user_id": user.id,
        "type": "topic_completion",
        "topic": topic,
        "course": course,
        "level": level,
        "date": datetime.now()
    })

    is_course_finished = False
    if not next_topic and curr_idx != -1:
         # Last topic?
         is_course_finished = True
         await history_collection.insert_one({
            "user_id": user.id,
            "type": "course_completion",
            "course": course,
            "level": level,
            "date": datetime.now()
         })

    return {
        "msg": "Topic completed", 
        "next_topic": next_topic,
        "progress": user_progress[course_key],
        "course_finished": is_course_finished
    }

# --- SEQUENTIAL COURSE LOGIC ---

def get_flattened_schedule(course: str, level: str):
    """
    Flattens the nested schedule into a list of topics.
    Returns: [{"topic": "...", "week": "...", "index": 0}, ...]
    """
    # Try fetching from DB first
    # This is synchronous but `course_collection` is async? 
    # Calling async from sync is tricky here. Let's make this async or use hardcode fallback for now.
    # To truly support dynamic, we should query `materials`.
    # For now, let's keep the hardcoded schedule for German level 1/2/etc if materials are empty.
    
    schedule = []
    
    # 1. Try DB Curriculum first
    from utils.mongodb import client
    # This is inside a synchronous call (maybe?) NO, get_flattened_schedule is sync but calls async?
    # Wait, get_flattened_schedule is defined as `def` (sync).
    # We cannot call `await` clearly here without changing to `async def`.
    # Let's verify usage. `get_course_structure` calls `flattened = get_flattened_schedule(course, level)`.
    # `complete_topic` calls it too.
    # We should refactor to async.
    pass

async def get_flattened_schedule_async(course: str, level: str):
    from utils.mongodb import curriculum_collection
    
    # Sanitize inputs
    course = course.strip() if course else course
    level = level.strip() if level else level

    # Case insensitive search using Regex
    cursor = curriculum_collection.find({
        "course": {"$regex": f"^{course}$", "$options": "i"}, 
        "level": {"$regex": f"^{level}$", "$options": "i"}
    }).sort("index", 1)


    print("cursor", cursor)
    items = await cursor.to_list(length=1000)
    print("items", items)
    print(f"DEBUG: get_flattened_schedule_async args: course='{course}', level='{level}'")
    print(f"DEBUG: Found {len(items)} items in DB")
    
    if not items:
        # Debugging Empty: Print what IS in the DB
        distinct_courses = await curriculum_collection.distinct("course")
        distinct_levels = await curriculum_collection.distinct("level")
        print(f"DEBUG: AVAILABLE COURSES IN DB: {distinct_courses}")
        print(f"DEBUG: AVAILABLE LEVELS IN DB: {distinct_levels}")

    if items:


        # print(f"Found {len(items)} items in DB for {course} {level}")
        return [
            {
                "index": i.get("index", idx), 
                "week": f"Module {i.get('index', idx) + 1}", 
                "topic": i.get("topic")
            }
            for idx, i in enumerate(items)
        ]


    # Fallback to Hardcoded if DB empty
    schedule = []
    if course == "German":
        level_data = GERMAN_SCHEDULE.get(level, {})
        idx = 0
        for week in sorted(level_data.keys()): 
             topics = level_data[week]
             for topic in topics:
                 schedule.append({
                     "index": idx,
                     "week": week,
                     "topic": topic
                 })
                 idx += 1
    return schedule


@app.get('/course_structure')
async def get_course_structure(user: UserOutput = Depends(decode_token)):
    """
    Returns the full course structure with status for each topic.
    """
    db_user = await get_user(user.id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
        
    course = db_user.get("enrolled_course", "").strip()
    level = db_user.get("enrolled_level", "").strip()
    print(f"Course: '{course}'")
    print(f"Level: '{level}'")

    
    # Default to 0 if not set
    current_index = db_user.get("current_topic_index", 0)
    
    if not course or not level:
         return {"structure": [], "message": "Not enrolled"}
         
    
    # 1. Get the ideal schedule
    flattened = await get_flattened_schedule_async(course, level)
    print("Flattened: ",flattened)
    
    # 2. Fetch uploaded materials
    from utils.mongodb import course_collection
    
    # Needs to match what uploadMaterial saves: "course_title", not "course"
    # Also case insensitive
    cursor = course_collection.find({
        "course_title": {"$regex": f"^{course}$", "$options": "i"}, 
        "level": {"$regex": f"^{level}$", "$options": "i"}
    })
    materials = await cursor.to_list(length=1000)
    print(f"DEBUG: Found {len(materials)} materials for {course} {level}")

    
    # 3. Create a lookup map: "Topic Name" -> Material Doc (Normalized keys)
    material_map = {m.get("topic").strip().lower(): m.get("file") for m in materials if m.get("topic")}
    material_cloud_map = {m.get("topic").strip().lower(): m.get("cloud_url") for m in materials if m.get("topic")}

    
    # 4. Merge
    final_structure = []
    
    # FETCH COMPLETED TOPICS FROM HISTORY (Source of Truth)
    from utils.mongodb import history_collection
    pipeline = [
        {"$match": {
            "user_id": user.id, 
            "type": "topic_completion",
            "course": course,
            "level": level
        }},
        {"$group": {"_id": "$topic"}}
    ]
    completed_cursor = history_collection.aggregate(pipeline)
    completed_list = await completed_cursor.to_list(length=1000)
    
    # Normalize for comparison
    completed_topics = {doc["_id"].strip().lower() for doc in completed_list if doc.get("_id")}

    for item in flattened:
        topic_name = item["topic"]
        # Normalize lookup
        topic_key = topic_name.strip().lower()
        
        has_material = topic_key in material_map
        is_completed = topic_key in completed_topics
        
        final_structure.append({
            **item,
            "has_material": has_material,
            "is_completed": is_completed,
            "file": material_map.get(topic_key),
            "cloud_url": material_cloud_map.get(topic_key)
        })

    print("Final Structure: ",final_structure)


    return {
        "structure": final_structure,
        "user_data": {
            "current_topic_index": current_index,
            "enrolled_course": course,
            "enrolled_level": level
        }
    }

@app.delete('/live_class/{id}')
async def delete_live_class(id: str, user: UserOutput = Depends(decode_token)):
    from utils.mongodb import live_classes_collection
    from bson.objectid import ObjectId
    
    try:
        res = await live_classes_collection.delete_one({"_id": ObjectId(id)})
        if res.deleted_count == 1:
            return {"msg": "Class deleted"}
        else:
            raise HTTPException(status_code=404, detail="Class not found")
    except Exception as e:
        print(f"Error deleting class: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete('/curriculum/{id}')
async def delete_curriculum_item(id: str, user: UserOutput = Depends(decode_token)):
    from utils.mongodb import curriculum_collection
    from bson.objectid import ObjectId
    
    try:
        res = await curriculum_collection.delete_one({"_id": ObjectId(id)})
        if res.deleted_count == 1:
            return {"msg": "Item deleted"}
        else:
            raise HTTPException(status_code=404, detail="Item not found")
    except Exception as e:
        print(f"Error deleting curriculum item: {e}")
        raise HTTPException(status_code=500, detail=str(e))






