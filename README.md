# 🧠 Iris – Context-Aware AI Research Paper Assistant (Flask + Ollama)

**Iris** is a subtle, elegant AI assistant built using Flask, Sentence Transformers, and LangChain.  
It supports **local LLM inference via Ollama** and context-based Q&A from PDFs.  
Designed for fast, local use without storing chat history.

---

## ⚙️ Features

- ✨ Clean UI with subtle design (no cringe)
- 📄 Upload research papers (PDF)
- 💬 Ask questions based on the content
- 🧠 Uses BGE embeddings + LangChain + Ollama
- 🛡️ No chat history saved. Fully local.
- 🌱 Works on mid-range laptops (Ryzen 5 + 1650 GPU tested)

---

## 🚀 Getting Started

### 1. Clone this repo

```bash
git clone https://github.com/ViperRyno/Iris-AI-Research-Companion
cd Iris-AI-Research-Companion
```

### 2. Create and activate a virtual environment

**Linux/macOS:**
```bash
python3 -m venv env
source env/bin/activate
```

**Windows:**
```bash
python -m venv venv
env\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Ollama locally

If you haven’t installed Ollama yet, follow [https://ollama.com](https://ollama.com) to install it.

Then run:

```bash
ollama run llama3:instruct
```

### 5. Run the Flask app

```bash
python app.py
```

The app will be available at:  
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📁 Project Structure

```
iris/
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
```

---

## 📝 License

This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgements

Built with ❤️ by Vicky using:

- [Flask](https://flask.palletsprojects.com/)
- [Ollama](https://ollama.com/)
- [LangChain](https://www.langchain.com/)
- `bge-large-en` Sentence Transformers from [SentenceTransformers](https://www.sbert.net/)
