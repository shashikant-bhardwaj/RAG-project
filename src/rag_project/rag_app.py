import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"

os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="RAG Chatbot", page_icon="📚")
st.title("📚 Chat with your Book")

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0


# ---------- Cached resources (loaded once) ----------
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings()

from langchain_groq import ChatGroq
@st.cache_resource
def get_llm_model():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.9,
    )


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


# ---------- Sidebar: upload + build vector store ----------
with st.sidebar:
    st.header("Upload your book")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Build Knowledge Base"):
            file_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Reading PDF, splitting into chunks, and creating embeddings..."):
                # load document
                data = PyPDFLoader(file_path)
                docs = data.load()

                # document splitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = splitter.split_documents(docs)

                # creating embeddings
                embedding_model = get_embedding_model()

                # creating vector store
                Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                    persist_directory=PERSIST_DIR
                )

            st.session_state.vectorstore_ready = True
            st.session_state.messages = []
            st.success(f"Knowledge base built from '{uploaded_file.name}'! You can start chatting now.")

    if st.session_state.vectorstore_ready:
        st.info("✅ Knowledge base is ready.")
    else:
        st.warning("⚠️ Upload a PDF and build the knowledge base to start chatting.")


# ---------- Main chat area ----------
if not st.session_state.vectorstore_ready:
    st.info("Upload a PDF book from the sidebar and click **Build Knowledge Base** to get started.")
else:
    embedding_model = get_embedding_model()
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )
    model = get_llm_model()

    # display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Ask something about your book...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        response = model.invoke(final_prompt)

        st.session_state.messages.append({"role": "assistant", "content": response.content})
        with st.chat_message("assistant"):
            st.write(response.content)