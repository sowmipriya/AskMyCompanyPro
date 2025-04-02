import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp

st.set_page_config(page_title='AskMyCompany Streamlit', layout='centered')
st.title('🧠 AskMyCompany Assistant (Streamlit)')

# Load vector DB
db = FAISS.load_local("faiss_index", SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)

llm = LlamaCpp(
    model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.7,
    max_tokens=512,
    n_ctx=2048,
    top_p=0.95,
    verbose=False,
)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

if "chat" not in st.session_state:
    st.session_state.chat = []

question = st.text_input("Ask a question about your company policies:")
if question:
    answer = qa.run(question)
    st.session_state.chat.append((question, answer))

for q, a in reversed(st.session_state.chat):
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
