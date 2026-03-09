import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

class ClassroomState(TypedDict):
    section_content: str

async def tutor_node(state: ClassroomState):
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, api_key=os.getenv('OPEN_AI_KEY'))
    messages = [
        SystemMessage(content="""You are an expert language tutor. 
Your goal is to explain the topic clearly and simply to the student as if you are teaching them naturally.
DO NOT mention that you are reading from a "material", "document", or "lesson plan". Speak directly to the student as their knowledgeable teacher.
Importantly, if the topic contains non-English words or phrases (such as German, French, etc.), you MUST explicitly translate them into English and explain their meaning, grammar, or context. 
Highlight key terms and ask one engaging check-in question at the end to ensure they understand.

CRITICAL FORMATTING RULE: You MUST format your response using proper Markdown. Use bolding (**word**), bulleted lists, and headers (###). You MUST use double spacing (two newlines) between every paragraph. Break concepts down into readable chunks. Never write a giant run-on wall of text."""),
        HumanMessage(content=f"Here is the topic you need to teach:\n{state['section_content']}")
    ]
    response = await llm.ainvoke(messages)
    return {"section_content": response.content} # We don't really need state updates since we only stream

workflow = StateGraph(ClassroomState)
workflow.add_node("tutor", tutor_node)
workflow.add_edge(START, "tutor")
workflow.add_edge("tutor", END)

classroom_app = workflow.compile()
