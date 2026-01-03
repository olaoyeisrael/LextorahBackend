from utils.mongodb import user_collection
async def get_user(user_id: str):
    user = await user_collection.find_one({"_id": user_id})
    if not user:
        user = {
            "_id": user_id,
            "file": None,
            "current_section": 1,
            "history_teach": [],
            "history_chat": [],
            "enrolled_course": None,
            "enrolled_level": None
        }
        await user_collection.insert_one(user)
    return user
