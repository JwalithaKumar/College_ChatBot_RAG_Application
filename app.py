
import streamlit as st
from rag_engine import load_knowledge_base, ask_rag

st.title("🤖 College Info Assistant")

@st.cache_resource
def initialize_rag():
    return load_knowledge_base("college_Docs")

embed_model, chunks, index = initialize_rag()

api_key = st.secrets["GROQ_API_KEY"]

# Initialize chat history
if "chat_hist" not in st.session_state:
    st.session_state.chat_hist = []

# Display previous messages
for msg in st.session_state.chat_hist:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Accept user question
question = st.chat_input("Ask about the college...")

if question:
    st.session_state.chat_hist.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Finding answer..."):
            answer = ask_rag(
                question, embedding_model, chunks, index,
                api_key=api_key, num_chunks=3
            )
            st.write(answer)

    st.session_state.chat_hist.append(
        {"role": "assistant", "content": answer}
    )

# Clear chat history
if st.button("Clear Chat"):
    st.session_state.chat_hist = []
    st.rerun()
