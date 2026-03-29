import streamlit as st
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

load_dotenv()

st.set_page_config(page_title="Medical Chatbot", page_icon="🏥", layout="centered")

# ── Custom CSS ──
st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
    }
    .stChatMessage {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 12px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    h1 {
        color: #38bdf8 !important;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        color: rgba(255,255,255,0.5);
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 Medical Chatbot")
st.markdown('<p class="subtitle">Ask me any health-related question</p>', unsafe_allow_html=True)


@st.cache_resource
def initialize_chain():
    embeddings = download_hugging_face_embeddings()
    index_name = "medical-chatbot"
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4, max_tokens=500)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


rag_chain = initialize_chain()

# ── Chat History ──
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── User Input ──
if user_input := st.chat_input("Type your health question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": user_input})
            answer = response["answer"]
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
