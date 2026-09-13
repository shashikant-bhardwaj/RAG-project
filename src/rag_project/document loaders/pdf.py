from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter


#  -------------------------------splitting based on token----------------------------------------
# spiltter = TokenTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap = 10
# )

# ---------------------------------recursively text spiltter---------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 1
) 
data = PyPDFLoader("data/interviewreact.pdf")

docs = data.load()

chunks = splitter.split_documents(docs)
for i in chunks:
    print(i.page_content)
    print()
    print()