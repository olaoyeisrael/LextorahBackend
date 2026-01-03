from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import os
import chromadb
from services.embedder import get_embeddings
from chromadb.config import Settings
from PyPDF2 import PdfReader
from services.chunker import chunk_text 
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from services.cloudinary_client import upload_image, upload_video, uploadMaterialToCloudinary
from utils.mongodb import course_collection
from utils.whisper import transcribe_audio
from datetime import datetime

load_dotenv()

# from langchain.vectorstores import Pinecone
# client = chromadb.CloudClient(
#   api_key='ck-6VA54Ws8DdwVkBzmoph69rcRCjCaKNGioehygzE8f8cA',
#   tenant='32dc770a-381e-4b0e-b9c1-91d1d88a790c',
#   database='lextorahmaterials'
# )
pc = Pinecone(api_key=os.getenv('PINECONE_API'))
index = pc.Index("lextorahdb")

def extract_text_from_file(filepath: str, filename: str):
    text = ""
    if filename.endswith(".pdf"):
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    elif filename.endswith((".mp3", ".wav", ".mp4")):
        text = transcribe_audio(filepath)

    return text





async def uploadMaterial(
    course_title: str = Form(...), 
    topic: str = Form(...),
    skill: str = Form(...),
    level: str = Form(...),
    file : UploadFile = File(...) 
) -> str:
    try:
        # 1. Save locally for teach_ws access
        os.makedirs("materials", exist_ok=True)
        # Use safe filename? The user said "topic is the material". 
        # But let's trust the file.filename for now, or use topic + extension?
        # User: "topic is 'adura' while material is adura.pdf"
        # It's safer to keep original filename but ensure it matches topic when searching/teaching.
        filepath = os.path.join("materials", file.filename)
        
        # Save directly from file object
        with open(filepath, "wb") as f:
            # We need to read it. But prompt says "await file.read()". 
            # If we read it here, we might consume it.
            # Best to read once.
            content = await file.read()
            f.write(content)
            
        # 2. Upload to Cloudinary (optional/backup)
        # We need to write content to a temp file for Cloudinary if needed, 
        # but we already have `filepath`.
        cloud_url = uploadMaterialToCloudinary(filepath)

        # 3. DB Entry
        course_doc = {
            "course_title" : course_title,
            "topic": topic,
            "skill": skill,
            "level": level,
            "filename": file.filename,
            "cloud_url": cloud_url,
            "content_type": file.content_type,
            "uploaded_at": datetime.now()
        }
        
        # Upsert based on topic/level? Or just insert? User might overwrite.
        # Let's just insert for now.
        inserted = await course_collection.insert_one(course_doc)
        course_id = str(inserted.inserted_id)

        # 4. Extract & Vectorize
        text = extract_text_from_file(filepath, file.filename)
        chunks = chunk_text(text)
        embeddings = []
        for chunk in chunks:

            chunk_embeddings = get_embeddings([chunk])
            embeddings.extend(chunk_embeddings)
        vectors = []
        for i, chunk in enumerate(chunks):
            vectors.append({
                "id": f"{course_id}_chunk_{i}",
                "values": embeddings[i],
                "metadata": {
                    "text": chunk,
                    "filename": file.filename,
                    "chunk_index": i,
                    "course": course_title,
                    "level": level,
                    "topic": topic
                }
            })
        index.upsert(vectors=vectors)
        # os.makedirs("materials", exist_ok=True)
        # filepath = os.path.join("materials", file.filename)
        # with open(filepath, "wb") as f:
        #     f.write(file.file.read())
        return "File uploaded and indexed successfully"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def retrieve_context(query: str, top_k: int = 5, filter: dict = None):
    query_vector = get_embeddings([query])[0]

    # print("query: ",query)
    # print("queryv: ",query_vector)
    # Ensure filter is handled if provided
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True, filter=filter)
    print("result: ", results)
    # Extract the text from metadata
    # context = [item['metadata']['text'] for item in results['matches']]
    context = [
        item['metadata']['text'] 
        for item in results['matches'] 
        if item['score'] >= 0.3
    ]
    print('context: ', context)
    return context



    # filepath = os.path.join("materials", file.filename)
    # with open(filepath, "wb") as f:
    #     f.write(file.file.read())
    # return "File uploaded successfully"


# def uploadMaterial(filename: str, filedata: bytes) -> str:
  
 
#     collection = client.get_or_create_collection(name="lextorahmaterials")
#     material_dir = "materials"
#     for filename in os.listdir(material_dir):
#         if filename.endswith(".pdf"):
#             filepath = os.path.join(material_dir, filename)
#             with open(filepath, "rb") as f:
#                 filedata = f.read()
#             # Extract text from the file (you can customize this part)
#             text = filedata.decode('utf-8', errors='ignore')
#             # Get embeddings
#             embeddings = get_embeddings(text)
#                     # Add to ChromaDB
#             collection.add(
#                 documents=['diamonds '],
#                 metadatas=[{"filename": 'sample.pdf'}],
#                 ids=['sample_id'],
#                 embeddings=embeddings
#             )
#     return "Materials uploaded and indexed successfully"




# def uploadMaterial(filename: str, contents: bytes) -> str:
#     os.makedirs("materials", exist_ok=True)
#     filepath = os.path.join("materials", filename)
#     with open(filepath, "wb") as f:
#         f.write(contents)
#     return "File uploaded successfully"