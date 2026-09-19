from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

search_tool = TavilySearchResults(
    max_result = 2
);

result = search_tool.invoke('ai news');
print(result)