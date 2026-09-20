import os
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
import requests
from langchain_core.tools import InjectedToolArg
from typing import Annotated

load_dotenv()

@tool
def get_conversion_factor(base_currency:str, target_currency:str)-> float:
    """
    This function fetches the currency conversion factor
    between a given base currency and a target currency.
    """
    url = f"https://v6.exchangerate-api.com/v6/2b3b93f90391db878060a7b9/pair/{base_currency}/{target_currency}"
    response = requests.get(url)

    return response.json()


result = get_conversion_factor.invoke({"base_currency":"USD", "target_currency":"INR"})

@tool
def convert(base_currency:int, conversion_rate:Annotated[float, InjectedToolArg])->float:
    """
    given a conversion rate this function calculate the target currency from a given base currency.
    """
    return base_currency*conversion_rate

result = convert.invoke({"base_currency":10, "conversion_rate":85})

# bind llm with tool

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    api_key = os.getenv("GROK_API_KEY")
)

llm_with_tool = llm.bind_tools([get_conversion_factor, convert])

query = HumanMessage("What is the conversion factor between USD and INR,  based on that can you convert 10 usd to inr")

Message = [query]

Ai_message = llm_with_tool.invoke(Message)

Message.append(Ai_message)

for tool_call in Ai_message.tool_calls:
    #Execute the 1st tool and get the value of conversion rate
 
    if tool_call["name"] == "get_conversion_factor":
        tool_message1 = get_conversion_factor.invoke(tool_call)

         #fetch this conversion rate
        conversion_rate = json.loads(tool_message1.content)["conversion_rate"]
        #append this tool to message list
        Message.append(tool_message1)
    
    
            
Ai_message2 = llm_with_tool.invoke(Message)
Message.append(Ai_message2)
    
for tool_call in Ai_message2.tool_calls:
    #Execute the second tool using the converion rate from tool1
    if tool_call["name"] == "convert":
        #fetch the current args
        tool_call["args"]["conversion_rate"] = conversion_rate
        tool_message2 = convert.invoke(tool_call)
        Message.append(tool_message2)  

final_result = llm_with_tool.invoke(Message)   
print(final_result.content)     
   


       








