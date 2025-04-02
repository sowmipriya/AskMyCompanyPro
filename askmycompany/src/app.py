from flask import Flask, request, render_template, session
import os, logging
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA

# LLM Switch
USE_OPENAI = False
USE_HUGGINGFACE = False

if USE_OPENAI:
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
elif USE_HUGGINGFACE:
    from langchain.llms import HuggingFaceHub
    llm = HuggingFaceHub(repo_id="google/flan-t5-base")
else:
    from langchain_community.llms import LlamaCpp
    llm = LlamaCpp(
        model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        temperature=0.7,
        max_tokens=512,
        n_ctx=2048,
        top_p=0.95,
        verbose=False,
    )

app = Flask(__name__)
app.secret_key = 'supersecretkey'
logging.basicConfig(filename='askmycompany.log', level=logging.INFO, format='%(asctime)s - %(message)s')

db = FAISS.load_local("faiss_index", SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

@app.route('/', methods=['GET', 'POST'])
def home():
    if 'history' not in session:
        session['history'] = []
    answer = ''
    if request.method == 'POST':
        question = request.form['question']
        answer = qa.run(question)
        session['history'].append((question, answer))
        session.modified = True
        logging.info(f'Q: {question} | A: {answer}')
    return render_template('index.html', answer=answer, history=session['history'])

if __name__ == '__main__':
    app.run(debug=True)
