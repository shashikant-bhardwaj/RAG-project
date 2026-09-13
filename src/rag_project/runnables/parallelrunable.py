import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



load_dotenv()

#1. two different prompt
short_prompt = ChatPromptTemplate.from_messages(
    "Explain {topic} in 1-2 lines"
)

detailed_prompt = ChatPromptTemplate.from_messages(
    "Explain {topic} in details"
)

#2, model
model = ChatGroq(
    model = "openai/gpt-oss-120b",
    api_key = os.getenv("GROQ_API_KEY"),
    temperature = 0.9
)

#3. parser
parser = StrOutputParser()