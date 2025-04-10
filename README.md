# 🧠 AskMyCompanyPro

AskMyCompanyPro is an intelligent, local-first document assistant built with **LangChain**, **FAISS**, **Flask**, and optionally **Streamlit**. It allows employees to query internal documents (e.g., PDF policies) and get instant, AI-powered answers using LLMs like **LlamaCpp**, **OpenAI**, or **Hugging Face**.

---

## 🚀 Features

- ✅ PDF document ingestion and chunking
- ✅ FAISS vector index creation
- ✅ Local or remote LLM support
- ✅ Flask Web UI with chat history
- ✅ Streamlit App option
- ✅ Multi-document support
- ✅ Docker and Docker Compose ready
- ✅ GitHub Actions workflow

---

## 🏗️ Project Structure

```
askmycompany/
├── data/                    # Drop PDF files here
├── models/                  # Local LLM model files
├── static/                  # CSS or frontend assets
├── templates/               # HTML templates for Flask
├── src/
│   ├── document_ingestion.py  # Load & embed PDFs
│   ├── app.py                 # Flask app with LLM switch
│   └── app_streamlit.py       # Streamlit version
├── utils/                   # Helper scripts
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ⚙️ LLM Options

| Mode         | Model Used                          | Switch        |
|--------------|-------------------------------------|----------------|
| 🧠 Local      | LlamaCpp (`.gguf`)                  | Default        |
| ☁️ OpenAI     | GPT-3.5 (`openai.ChatOpenAI`)       | `USE_OPENAI`   |
| 🤗 HuggingFace | FLAN-T5 or other HF models         | `USE_HUGGINGFACE` |

Edit these flags in `app.py` to change your LLM.

---

## 📦 Install Locally (Dev Mode)

```bash
git clone https://github.com/sowmipriya/AskMyCompanyPro.git
cd AskMyCompanyPro
pip install -r requirements.txt

# Optional: Build vector DB
python askmycompany/src/document_ingestion.py

# Run Flask app
python askmycompany/src/app.py
```

---

## 🐳 Docker Usage

```bash
# One-liner with Docker Compose
docker-compose up --build
```

Or manually:

```bash
docker build -t askmycompany .
docker run -p 8501:8501 askmycompany
```

---

## 🧪 Run Streamlit Version

```bash
streamlit run askmycompany/src/app_streamlit.py
```

---

## 🔐 GitHub Actions CI/CD

Auto-deploy on every `main` push via `.github/workflows/deploy.yml`.

---

