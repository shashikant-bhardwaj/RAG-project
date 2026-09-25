import os
from dotenv import load_dotenv
import requests
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from rich import print



load_dotenv()

"""
# creating tools
# -> Weather tool
@tool
def get_weather(city: str) -> str:
   #yeh doc string hai
  #  get current weather of a given city
    
    WEATHER_API = os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API}&units=metric"
    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        return f"Error: {data.get('message', 'could not fetch weather')}"

    temp = data["main"]["temp"]
    description = data["weather"][0]["description"]
    name = data["name"]

    return (
        f"City: {name}\n"
        f"Temperature: {temp}degree celsius\n"
        f"Weather: {description}\n"
    )


# print(get_weather.invoke({"city": "Shimla"}))


# -> Tavily news tool

TAVILY_API = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(TAVILY_API)


@tool
def get_news(city: str) -> str:
    #yaha doc string aayegi
   # get latest news about the city
    
    response = tavily_client.search(
        query=f"latest news in {city}", search_depth="basic", max_results=3
    )

    results = response.get("results", [])

    if not results:
        return f"No latest news found for {city}"

    news = []

    for index, result in enumerate(results, start=1):
        title = result.get("title", "No title")
        content = result.get("content", "No content")
        url = result.get("url", "")

        news.append(f"{index}: {title}\n" f"content: {content}\n" f"url: {url}\n")

    return f"Latest news in {city}:\n\n" + "\n\n".join(news)



# creating llm

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROK_API_KEY"))

tools = {"get_weather": get_weather, "get_news": get_news}

# binding llm with tool
llm_with_tool = llm.bind_tools([get_weather, get_news])


# AGENT LOOP - VERY IMPORTANT

Message = []

print("City Intelligence System")
print("Type 0 to Quit")

while True:
    user_input = str(input("You : "))
    if user_input.lower() == "exit":
        break

    query = HumanMessage(content=user_input)

    Message.append(query)

    while True:
        AI_Message = llm_with_tool.invoke(Message)

        Message.append(AI_Message)

        # if tool is required
        if AI_Message.tool_calls:
            for tool_call in AI_Message.tool_calls:
                tool_name = tool_call["name"]

                # HUMAN IN THE LOOP
                confirm = input(f"Agent wants to call {tool_name} Approve (yes/no)")

                if confirm.lower() != "yes":
                    print("tool call denied and I cannot get the latest information")
                    break

                # Execute tool
                tool_result = tools[tool_name].invoke(tool_call)

                Message.append(
                    ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
                )

            continue
        else:
            print("\n Final Answer: \n")
            print(AI_Message.content)    
            break;

"""

#-------------------part 2 creating agent with langchain create_agent------------------------
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call

# creating tools
# -> Weather tool
@tool
def get_weather(city: str) -> str:
    """
    get current weather of a given city
    """
    WEATHER_API = os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API}&units=metric"
    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        return f"Error: {data.get('message', 'could not fetch weather')}"

    temp = data["main"]["temp"]
    description = data["weather"][0]["description"]
    name = data["name"]

    return (
        f"City: {name}\n"
        f"Temperature: {temp}degree celsius\n"
        f"Weather: {description}\n"
    )


# print(get_weather.invoke({"city": "Shimla"}))


# -> Tavily news tool

TAVILY_API = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(TAVILY_API)


@tool
def get_news(city: str) -> str:
    """
    get latest news about the city
    """
    response = tavily_client.search(
        query=f"latest news in {city}", search_depth="basic", max_results=3
    )

    results = response.get("results", [])

    if not results:
        return f"No latest news found for {city}"

    news = []

    for index, result in enumerate(results, start=1):
        title = result.get("title", "No title")
        content = result.get("content", "No content")
        url = result.get("url", "")

        news.append(f"{index}: {title}\n" f"content: {content}\n" f"url: {url}\n")

    return f"Latest news in {city}:\n\n" + "\n\n".join(news)


llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROK_API_KEY"))

@wrap_tool_call
def human_approval(request, handler):
    """
    Ask for human approval before every tool call.
    """
    tool_name = request.tool_call["name"]
    confirm = input(f"Agent  want to call '{tool_name}', Approve (yes/no): ")

    if confirm.lower() != "yes":
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id = request.tool_call["id"]
        )

    return handler(request)

agent = create_agent(
    llm,
    tools = [get_weather, get_news],
    system_prompt= "you are a helpful city assistant",
    middleware=[human_approval]
)

print("City Agent | Type exit to quit")

while True:
    user_input = input("You : ")
    if user_input.lower() == "exit":
        break;

    result = agent.invoke({
       "messages": [{"role": "user", "content": user_input}]
   })

    #print("Bot : ", result['messages'][-1].content)

    print(result);

    