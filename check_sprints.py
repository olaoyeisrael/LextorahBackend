import asyncio
from utils.mongodb import sprints_collection
async def main():
    doc = await sprints_collection.find_one({})
    print(doc)
asyncio.run(main())
