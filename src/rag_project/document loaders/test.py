from langchain_community.document_loaders import TextLoader

data = TextLoader("data/interviewreact.txt")

docs = data.load()

print(docs[0])