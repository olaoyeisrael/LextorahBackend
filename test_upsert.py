import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.getenv('PINECONE_API'))
index = pc.Index("lextorahdb")
try:
    index.upsert(vectors=[])
    print("Success")
except Exception as e:
    print(f"Error: {e}")
