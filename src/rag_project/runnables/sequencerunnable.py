import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
#1. prompt template 
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

#2. model
model = ChatGroq(
    model = "openai/gpt-oss-120b",
    api_key= os.getenv("GROQ_API_KEY"),
    temperature=0.9
)

#3. output parser
parser = StrOutputParser()


# #step by step manual flow

# #format the prompt
# formatted_prompt = prompt.format_messages(topic = "Machine learning")

# #call the model manually
# response = model.invoke(formatted_prompt)

# #parse the output manually
# final_prompt = parser.parse(response.content)

# print(final_prompt)

chain = prompt | model | parser

result = chain.invoke("Machine learning")

print(result)