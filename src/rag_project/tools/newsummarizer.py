import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
search_tool = TavilySearchResults(
    max_result = 3
)

prompt = ChatPromptTemplate.from_template(
    """
you are a helpful assistant 

summarizes the following news into the  clear bullet points
{news}
    """
)

model = ChatGroq(
    model = "openai/gpt-oss-120b",
    api_key = os.getenv("GROQ_API_KEY")
)

parser = StrOutputParser()

chain = prompt | model | parser

news_result = search_tool.run("Latest AI news of hacking hugging face of 2026")

result = chain.invoke({"news": news_result})

print(result)




