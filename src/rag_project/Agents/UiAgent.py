import os
import streamlit as st
from dotenv import load_dotenv
import requests
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient

load_dotenv()

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call

# ---------- creating tools ----------
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


# -> Tavily news tool
@st.cache_resource
def get_tavily_client():
    return TavilyClient(os.getenv("TAVILY_API_KEY"))


tavily_client = get_tavily_client()


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


# ---------- llm (cached so it's not re-created on every rerun) ----------
@st.cache_resource
def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROK_API_KEY"))


llm = get_llm()


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
            tool_call_id=request.tool_call["id"]
        )

    return handler(request)


@st.cache_resource
def get_agent():
    return create_agent(
        llm,
        tools=[get_weather, get_news],
        system_prompt="you are a helpful city assistant",
        middleware=[human_approval]
    )


agent = get_agent()

# ---------- UI ----------
st.set_page_config(page_title="City Agent", page_icon="🏙️", layout="wide")

with st.sidebar:
    st.header("🏙️ City Agent")
    st.caption("Ask about weather or the latest news in any city.")
    st.divider()
    st.markdown("**Tools available:**")
    st.markdown("- ☁️ Weather lookup")
    st.markdown("- 📰 Latest news")
    st.divider()
    st.info("⚠️ Tool calls require approval in the terminal where this app was launched.")
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🏙️ City Agent")
st.caption("Your assistant for city weather and news updates")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

user_input = st.chat_input("Ask about a city's weather or news...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.write(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking... (check your terminal if tool approval is needed)"):
            result = agent.invoke({
                "messages": [{"role": "user", "content": user_input}]
            })
            bot_content = result["messages"][-1].content
        st.write(bot_content)

    st.session_state.messages.append({"role": "assistant", "content": bot_content})