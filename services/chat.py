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
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.7,
    api_key=os.getenv('OPEN_AI_KEY')
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are Ms Lexi, a helpful AI tutor who uses the Socratic method to guide students.\n"
        "Your goal is to help the user discover the correct answer on their own through questioning and reasoning, rather than giving the answer away directly.\n"
        "Guidelines:\n"
        "1. Never give a direct answer, full summary, or direct solution to the user's questions.\n"
        "2. Carefully read the provided Context to understand the facts, but do not state the context or repeat facts directly to the student.\n"
        "3. Instead, ask guiding, thought-provoking, and open-ended questions that prompt the student to think critically and take the next step.\n"
        "4. Break down complex concepts into small, manageable parts. Focus on one step/concept at a time.\n"
        "5. Validate the user's correct reasoning and gently guide them back when they make a mistake by asking a clarifying or simpler question.\n"
        "6. Always keep your response concise, polite, and educational."
    ),
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
        # Get list of enrolled course codes (supporting 'enrolled_courses', 'courses', or 'course_codes')
        enrolled_codes = user.get("enrolled_courses") or user.get("courses") or user.get("course_codes") or []
        
        # Normalize to list
        if isinstance(enrolled_codes, str):
            enrolled_codes = [enrolled_codes]
        elif not isinstance(enrolled_codes, list):
            enrolled_codes = []
            
        # Fallback to single enrolled_course string if not already present
        single_course = user.get("enrolled_course")
        if single_course and single_course not in enrolled_codes:
            enrolled_codes.append(single_course)
            
        # Clean up strings
        enrolled_codes = [c.strip() for c in enrolled_codes if c and isinstance(c, str)]
        
        # Build Pinecone metadata filter based on course codes
        if enrolled_codes:
            if len(enrolled_codes) == 1:
                filter = {
                    "course": {"$eq": enrolled_codes[0]}
                }
            else:
                filter = {
                    "course": {"$in": enrolled_codes}
                }
        
        print(f"DEBUG: retrieve_context filter for user '{user_id}' with enrolled course codes {enrolled_codes}: {filter}")

    
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

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader

from langchain_core.tools import tool

@tool
def load_web_page(url: str) -> str:
    """Load and read a specific page from Lextorah or partner websites to find answers to customer support queries.
    Allowed websites/domains: www.homeworks.ng, www.lextorah-elearning.com, www.lextorah.com, www.lextorahsolutions.com.ng, www.lextorahjobs.com, www.Lextorah.aI
    """
    try:
        # Prepend scheme if missing
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url

        # Normalize the URL for domain check
        url_lower = url.lower()
        allowed_domains = [
            "homeworks.ng",
            "lextorah-elearning.com",
            "lextorah.com",
            "lextorahsolutions.com.ng",
            "lextorahjobs.com",
            "lextorah.ai"
        ]
        
        # Check if the URL domain is allowed
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname or ""
        
        is_allowed = False
        for domain in allowed_domains:
            if hostname == domain or hostname.endswith("." + domain):
                is_allowed = True
                break
                
        if not is_allowed:
            return f"Please only load URLs from the allowed domains: {', '.join(allowed_domains)}."
            
        loader = WebBaseLoader(url)
        docs = loader.load()
        content = docs[0].page_content if docs else ""
        print(f"Loaded content from {url}: {content[:200]}...")  # Log first 200 characters for debugging
        return content[:4000]
    except Exception as e:
        print(f"Load error: {e}")
        return f"Failed to load the web page: {e}"



@tool
def read_support_script(query: str = "") -> str:
    """Read the Lextorah support script PDF to find information about booking language classes, tuition fees, and registration steps."""
    try:
        loader = PyPDFLoader("materials/support_script.pdf")
        docs = loader.load()
        content = "\n".join([doc.page_content for doc in docs])
        return content
    except Exception as e:
        print(f"PDF Load error: {e}")
        return "Failed to load the support script."

support_tools = [read_support_script, load_web_page]
support_agent = create_agent(model=llm, tools=support_tools, system_prompt=("You are Ms Lexi, a helpful customer support agent for Lextorah. Use the read_support_script tool to get step-by-step processes and booking info from the local PDF script, and use the load_web_page tool to fetch content from Lextorah and partner sites: www.homeworks.ng, www.lextorah-elearning.com, www.lextorah.com, www.lextorahsolutions.com.ng, www.lextorahjobs.com, and www.Lextorah.aI. You must STRICTLY restrict your answers to topics related to Lextorah, its partner sites, and their services. If a user asks a question entirely unrelated, politely decline to answer. Be polite, concise, and helpful."), checkpointer=MemorySaver())


async def chat_support(session_id: str, question: str):
    response = support_agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"thread_id": session_id}}
        )

    final_response = response.get("messages")[-1].content if response.get("messages") else "Sorry, I couldn't find an answer to your question."
    return final_response
