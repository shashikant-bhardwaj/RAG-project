import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
# from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import  ChatPromptTemplate
load_dotenv()

#  -------------------------loading document ------------------------------------

# data = TextLoader("data/interviewreact.txt")
data = PyPDFLoader("data/interviewreact.pdf")
docs = data.load()
template = ChatPromptTemplate.from_messages(
    [("system", "you are a AI that summarize text"),
     ("human", "{data}")]
)

llm = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
model = ChatHuggingFace(llm=llm)

prompt = template.format_messages(data = docs[0].page_content)


response = model.invoke(prompt)

print(response.content)