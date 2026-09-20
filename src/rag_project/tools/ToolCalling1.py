import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool

load_dotenv()


@tool
def multiply(a:int, b:int)-> int:
    "Multiply the given two numbers"
    return a*b

llm = ChatGroq(
    model= "openai/gpt-oss-120b",
    api_key= os.getenv("GROK_API_KEY")
)

#binding tool with llm
llm_with_tool = llm.bind_tools([multiply])

query = HumanMessage("can you multiply 3 with 20")
Messages = [query]

# print(Messages)

result = llm_with_tool.invoke(Messages)
Messages.append(result)

# print(Messages)

tool_result = multiply.invoke(result.tool_calls[0])
Messages.append(tool_result)

# print(Messages)

final_result = llm_with_tool.invoke(Messages)

print(final_result.content)


