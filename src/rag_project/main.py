import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()

embedding_model = HuggingFaceEmbeddings()

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Flash-0731",
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(
    llm = llm
)
# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """you are a helpful AI assistant.
            Use only the provided context to answer the question,
                 
           if the answer is not present in the context,
           say: "it could not find the answer in the document"""),
        ("human",
         """context: {context}
            Question: {question}
          """)
    ]
)
print("Rag system created")
print("press 0 to exit")

while True:

    query = input("You : ")

    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = model.invoke(final_prompt)

    print(f"\nAI: {response.content}")