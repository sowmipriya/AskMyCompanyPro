from flask import Flask, request, render_template, session
from PyPDF2 import PdfReader
import os, logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# --- App Configuration ---
app = Flask(__name__)
app.secret_key = 'supersecretkey'
logging.basicConfig(filename='chatpdf.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# --- Global Constants ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_PATH = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
VECTOR_INDEX_PATH = "faiss_index"

# --- Core Logic Functions ---

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local(VECTOR_INDEX_PATH)

def load_vector_store():
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(VECTOR_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context,
    just say "answer is not available in the context". Don't guess or provide wrong information.

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    llm = LlamaCpp(
        model_path=LLM_PATH,
        temperature=0.3,
        max_tokens=512,
        n_ctx=2048,
        top_p=0.95,
        verbose=False
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

def get_answer(question):
    db = load_vector_store()
    docs = db.similarity_search(question)
    chain = get_conversational_chain()
    result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return result["output_text"]

# --- Routes ---

@app.route('/', methods=['GET', 'POST'])
def home():
    if 'history' not in session:
        session['history'] = []

    answer = ''
    if request.method == 'POST':
        if 'question' in request.form:
            question = request.form['question']
            answer = get_answer(question)
            session['history'].append((question, answer))
            session.modified = True
            logging.info(f"Q: {question} | A: {answer}")
        elif 'process_pdfs' in request.form:
            files = request.files.getlist("pdfs")
            raw_text = get_pdf_text(files)
            chunks = get_text_chunks(raw_text)
            get_vector_store(chunks)
            answer = "PDFs processed and vector store created."

    return render_template('index.html', answer=answer, history=session['history'])

# --- Run ---
if __name__ == '__main__':
    app.run(debug=True)
