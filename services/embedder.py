from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv('OPEN_AI_KEY')
)

def get_embeddings(filetexts):
    embedding = embeddings.embed_documents(filetexts)
    print("Embeddings generated:", embedding)
    return embedding