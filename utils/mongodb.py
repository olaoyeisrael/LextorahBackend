from pymongo import MongoClient
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI)
db = client["lextorahmaterials"]
course_collection = db["courses"]
user_collection = db["users"]
data_collection = db["data"]
curriculum_collection = db["curriculums"]
user_course_collection = db["user_courses"]
quiz_results_collection = db["quiz_results"]
live_classes_collection = db["live_classes"]
history_collection = db["history"]