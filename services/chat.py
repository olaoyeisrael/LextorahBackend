import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from services.uploadMaterial import retrieve_context
from utils.helper import get_user
from utils.mongodb import user_collection

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.7,
    api_key=os.getenv('OPEN_AI_KEY')
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are Ms Lexi, a helpful assistant. Use the context below to answer the user's question."),
    MessagesPlaceholder(variable_name="history_messages"),
    HumanMessagePromptTemplate.from_template("Question: {question}\nContext:\n{context}"),
])



chain = prompt | llm | StrOutputParser()

store = {}

async def get_session_history(config):
    
    user_id = config["configurable"]["session_id"]
    # Handle composite IDs like "userid-chat" if necessary
   
    user = await get_user(user_id)
    if not user:
         # Handle case where user is not found, maybe return empty history or create temp
         return ChatMessageHistory()

    return mongo_history_to_lc(user.get("history_chat", []))

def mongo_history_to_lc(history_list):
    history = ChatMessageHistory()
    for msg in history_list:
        role = msg.get("role")
        content = msg.get("content", "")
        if not content:
            continue  # skip empty messages
        
        if role == "user":
            history.add_user_message(content)
        elif role in ("assistant", "ai"):
            history.add_ai_message(content)
        else:
            # optional: skip or log unknown roles
            continue
        print("History:",history)
    return history




# def chat_with_model (chatid: str, question: str):
#     context = retrieve_context(question)
#     context_text = "\n".join(context) if context else "No relevant context found."
#     chat_with_history = RunnableWithMessageHistory(
#         chain,
#         get_session_history,
#         input_messages_key="question",
#         history_messages_key="history_messages"
#     )
#     res = chat_with_history.invoke(
#         {"question": question, "context": context_text},
#         {"configurable": {"session_id": chatid}}
#     )
#     print("Store after invoke:", store)
#     return res


async def chat_with_model(user_id: str, question: str):
    # Ensure user_id is extracted correctly first to get user filter
    user = await get_user(user_id)
    
    filter = None
    if user:
         course = user.get("enrolled_course")
         level = user.get("enrolled_level")
         
         if course and level:
             filter = {
                 "course": {"$eq": course},
                 "level": {"$eq": level}
             }
         elif course:
             filter = {"course": {"$eq": course}}

    
    context = retrieve_context(question, filter=filter)
    context_text = "\n".join(context) if context else "No relevant context found."
    
    user = await get_user(user_id)
    if not user:
        return "User not found."

    print(f"Chat session: {user_id}")

    history_key = "history_chat"

    print("history key: ", history_key)

    # Load history from Mongo
    lc_history = mongo_history_to_lc(user.get(history_key, []))
    print("Lc History",lc_history)

    chat_with_history = RunnableWithMessageHistory(
        chain,
        lambda config: lc_history,
        input_messages_key="question",
        history_messages_key="history_messages"
    )

    try:
        print("Invoking chain...")
        msg = await chat_with_history.ainvoke(
            {"question": question, "context": context_text},
            {"configurable": {"session_id": user_id}}
        )
        print(f"Chain response: {msg}")
    except Exception as e:
        print(f"Chain invocation failed: {e}")
        return "I'm having trouble thinking right now."

    # # Append user + assistant messages to Mongo
    if history_key not in user:
        user[history_key] = []
    
    # Update local object (optional, but good for consistency if reused)
    user[history_key].append({"role": "user", "content": question})
    user[history_key].append({"role": "assistant", "content": msg})

    await user_collection.update_one(
        {"_id": user_id},
        {"$push": {history_key: {"$each": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": msg}
        ]}}}
    )

    print("Updated history for user:", user_id)
    return msg


# In-memory store for anonymous sessions
in_memory_store = {}

def get_in_memory_history(session_id: str) -> ChatMessageHistory:
    if session_id not in in_memory_store:
        in_memory_store[session_id] = ChatMessageHistory()
    return in_memory_store[session_id]

from duckduckgo_search import DDGS

def search_lextorah(query: str):
    try:
        results = DDGS().text(f"site:lextorah.com {query}", max_results=3)
        return [f"Title: {r['title']}\nSnippet: {r['body']}\nLink: {r['href']}" for r in results]
    except Exception as e:
        print(f"Search error: {e}")
        return []

async def chat_support(session_id: str, question: str):
    # Retrieve context to help answer questions about the platform
    # Combine vector store context with live web search
    web_context = search_lextorah(question)
    
    combined_context = web_context
    context_text = "\n\n".join(combined_context) if combined_context else "No relevant context found."

    chat_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: get_in_memory_history(session_id),
        input_messages_key="question",
        history_messages_key="history_messages"
        
    )

    msg = await chat_with_history.ainvoke(
        {"question": question, "context": context_text},
        {"configurable": {"session_id": session_id}}
    )
    return msg

