import sys
from services.model import llm
from langchain_core.messages import HumanMessage

msg = HumanMessage(content="Reply with OK")
try:
    print(llm.invoke([msg]))
except Exception as e:
    print("Error:", e)
