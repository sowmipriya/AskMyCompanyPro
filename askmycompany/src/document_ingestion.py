from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
import os

def load_all_pdfs(directory="askmycompany/data"):
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(directory, filename))
            pages = loader.load()
            docs = splitter.split_documents(pages)
            all_docs.extend(docs)
    return all_docs

def build_vector_index():
    docs = load_all_pdfs()
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embedding_model)
    db.save_local("faiss_index")
    print(" Vector store created from all PDFs.")

if __name__ == "__main__":
    build_vector_index()
