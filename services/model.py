import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.7,
    api_key=os.getenv('OPEN_AI_KEY')
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
    "You teach students in short, simple, direct sentences. "
    "You never summarize. "
    "You never use headings, lists, bullet points, or sections. "
    "You never add closing remarks. "
    "You never mention context, previous messages, chat history, memory, or any database. "
    "You never say phrases like 'based on the context', 'from your message', or 'according to what you provided'. "
    "You speak as if you already understand the material without referencing why. "
    "You only give a short, natural explanation like a teacher talking directly to a student."
),
    MessagesPlaceholder(variable_name="history_messages"),
    HumanMessagePromptTemplate.from_template("{question}"),
])

chain = prompt | llm | StrOutputParser()

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
        print("New session added. Store now:", store)
    return store[session_id]

def ask_with_history(chatid: str, question: str):
    chat_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history_messages"
    )
    res = chat_with_history.invoke(
        {"question": question},
        {"configurable": {"session_id": chatid}}
    )
    print("Store after invoke:", store)
    return res