import asyncio
from utils.mongodb import quiz_results_collection

async def main():
    doc = await quiz_results_collection.find_one()
    print("Sample Quiz Result Document:")
    print(doc)

if __name__ == "__main__":
    asyncio.run(main())
