import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from rich import print
from langchain_core.messages import HumanMessage

load_dotenv()

#--------------------------- part-1 ----------------------------------------
# 1 creating a tool
@tool
def get_text_length(text: str) -> int:
    """Returns the number of character in a given text"""
    return len(text)


tools = {
    "get_text_length": get_text_length
}


# LLM
llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROK_API_KEY"))

# TOOL BINDING

llm_with_tool = llm.bind_tools([get_text_length])

Message = []

while True:

 question = str(input("You : "))

 query = HumanMessage(question)

 Message.append(query)

 AI_message = llm_with_tool.invoke(Message)

 Message.append(AI_message)

 if not AI_message.tool_calls:
    print("Bot : ", AI_message.content)
    continue;

 if AI_message.tool_calls:
    tool_name = AI_message.tool_calls[0]["name"]
    tool_messsage = tools[tool_name].invoke(AI_message.tool_calls[0])
    Message.append(tool_messsage)


 final_result = llm_with_tool.invoke(Message)

 print("Bot : ", final_result.content)


