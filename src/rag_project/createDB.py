# load the pdf
# spilt into chunks
# create embeddings
# store into chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv


load_dotenv()

# load document 
data = PyPDFLoader("data/deep-learning-material.pdf")
docs = data.load()

# document splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks  = splitter.split_documents(docs)

# creating embeddings
embedding_model = HuggingFaceEmbeddings()

# creating vector store
vectorStore = Chroma.from_documents(
    documents= chunks,
    embedding= embedding_model,
    persist_directory= "chroma_db"
)

