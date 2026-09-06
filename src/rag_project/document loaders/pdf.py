from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("data/interviewreact.pdf")

docs = data.load()

print(len(docs))