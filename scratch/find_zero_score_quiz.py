import asyncio
from utils.mongodb import quiz_results_collection

async def main():
    cursor = quiz_results_collection.find()
    docs = await cursor.to_list(length=100)
    print(f"Total documents: {len(docs)}")
    for idx, doc in enumerate(docs):
        print("-" * 50)
        print(f"Doc {idx}: ID={doc['_id']}, User={doc.get('user_id')}, Topic={doc.get('topic')}, Score={doc.get('score')}/{doc.get('total')}")
        print(f"Details count: {len(doc.get('details', [])) if doc.get('details') else 'None'}")
        if doc.get("details"):
            print("First 2 details:")
            for d in doc["details"][:2]:
                print(f"  Q: {d.get('question')}")
                print(f"  Ans: {d.get('answer') or d.get('user_answer')}")
                print(f"  Correct: {d.get('correct') or d.get('correct_answer') or d.get('correct_option')}")

if __name__ == "__main__":
    asyncio.run(main())
