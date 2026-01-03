import asyncio
from utils.mongodb import curriculum_collection, db as main_db

async def check():
    print("Checking database:", main_db.name)
    print("Collection:", curriculum_collection.name)
    
    count = await curriculum_collection.count_documents({})
    print(f"Total items in curriculums: {count}")
    
    items = await curriculum_collection.find({}).to_list(length=10)
    for i in items:
        print(i)
        
    # Check distinct courses/levels
    courses = await curriculum_collection.distinct("course")
    levels = await curriculum_collection.distinct("level")
    print("Courses in DB:", courses)
    print("Levels in DB:", levels)

if __name__ == "__main__":
    asyncio.run(check())
