import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from rich import print

load_dotenv()

#--------------------------- part-1 ----------------------------------------
# 1 creating a tool
# @tool
# def get_text_length(text: str) -> int:
#     """Returns the number of character in a given text"""
#     return len(text)


# # LLM
# llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROK_API_KEY"))

# # TOOL BINDING

# llm_with_tool = llm.bind_tools([get_text_length])


# result = llm.invoke("Returns the number of character in a given text: 'Hello how are you'")
# result2 = llm_with_tool.invoke("Returns the number of character in a given text: Hello how are you")


# print(result)
# print()
# print()
# print()
# print()
# print(result2)

# note:- jab bhi aap llm ke sath tool bind krte hai -->
# toh jab bhi uss llm ko invoke kroge toh token increase ho jayenge-->
#  kyuki jo tool tumne bnayah h uske andr doc string ya sab meta deta token mei count hoga.







#-------------------------- part-2 ------------------------------------

# 1 creating a tool
@tool
def get_text_length(text: str) -> int:
    """Returns the number of character in a given text"""
    return len(text)


# LLM
llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROK_API_KEY"))

# TOOL BINDING

llm_with_tool = llm.bind_tools([get_text_length])


# step1:  LLM decides tool
result = llm_with_tool.invoke(
    "Use get_text_length tool to find the length of: Hello how are you"
)


# step2-4: Execute tool
if result.tool_calls:
    tool_call = result.tool_calls[0]
    tool_result = get_text_length.invoke(tool_call["args"])

# step5: send back to LLM
final_response = llm.invoke(f"the length of text result is {tool_result}")
print(final_response.content)








#-------------------------- part-3 ------------------------------------

# 1 creating a tool
@tool
def get_text_length(text: str) -> int:
    """Returns the number of character in a given text"""
    return len(text)


# LLM
llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROK_API_KEY"))

# TOOL BINDING

llm_with_tool = llm.bind_tools([get_text_length])


# step1:  LLM decides tool
result = llm_with_tool.invoke(
    "Use get_text_length tool to find the length of: Hello how are you"
)

print(result.tool_calls[0])

print(get_text_length.invoke({
    'name': 'get_text_length',
    'args': {'text': 'Hello how are you'},
    'id': 'fc_ebaae4fd-6b44-40b6-8dd5-71c6fc3a3e41',
    'type': 'tool_call'
}))
