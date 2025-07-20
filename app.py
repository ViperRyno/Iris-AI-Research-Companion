from flask import Flask, request, render_template, session, redirect, url_for
import os
import fitz
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import shutil
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

app = Flask(__name__)
app.secret_key = 'secret123'

UPLOAD_FOLDER = 'uploads'
if os.path.exists(UPLOAD_FOLDER):
    shutil.rmtree(UPLOAD_FOLDER)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Embedding model
embedding_function = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-large-en-v1.5")
chroma_client = chromadb.Client(settings=chromadb.config.Settings(allow_reset=True))

# LLM setup
llm = Ollama(model="llama3:instruct")

# In-memory metadata only
global_state = {}

# --------------------- Utility Functions ---------------------

def chunk_text(text, chunk_size=150, overlap=30):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def extract_text_sections(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])
    sections = re.split(
        r"\n(?=(Abstract|Introduction|Related Work|Methodology|Experiments|Results|Conclusion|References))",
        full_text, flags=re.IGNORECASE)
    section_dict = {}
    for i in range(1, len(sections), 2):
        title = sections[i].strip().lower()
        content = sections[i + 1].strip()
        section_dict[title] = content
    return section_dict

def create_chroma_collection(pdf_id, chunks, section):
    collection = chroma_client.get_or_create_collection(name=pdf_id, embedding_function=embedding_function)
    for i, chunk in enumerate(chunks):
        if chunk.strip():
            collection.add(
                documents=[chunk],
                ids=[f"{pdf_id}_chunk_{section}_{i}"],
                metadatas=[{'section': section, 'chunk_id': i}]
            )

def process_pdf(pdf_path, pdf_id):
    sections = extract_text_sections(pdf_path)
    for section, text in sections.items():
        chunks = chunk_text(text)
        create_chroma_collection(pdf_id, chunks, section)

def retrieve_chunks_chroma(query, collection_name, k=10):
    collection = chroma_client.get_collection(name=collection_name)
    results = collection.query(query_texts=[query], n_results=k)
    return results['documents'][0]

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Optional: basic custom prompt if you still want it (else skip this)
qa_prompt = PromptTemplate.from_template(
    "You are an expert academic assistant helping users understand and analyze research papers.\n"
    "Use the following extracted context from one or more papers to answer the question clearly and concisely.\n"
    "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
)

def generate_answer(query, chunks):
    docs = [Document(page_content=chunk) for chunk in chunks]
    chain = load_qa_chain(llm, chain_type="stuff",prompt=qa_prompt)  # No prompt needed unless you want customization
    return chain.run(input_documents=docs, question=query)

# --------------------- Flask Routes ---------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    chat_history = session.get('chat_history', [])
    uploaded_files = session.get('uploaded_files', [])
    pdf_ids = session.get('pdf_ids', [])
    uploaded = bool(uploaded_files)

    if request.method == 'POST':
        if 'pdfs' in request.files:
            pdf_list = request.files.getlist('pdfs')

            for i, pdf in enumerate(pdf_list, start=len(uploaded_files) + 1):
                filename = pdf.filename
                pdf_id = f"paper_{i}"
                path = os.path.join(UPLOAD_FOLDER, filename)
                pdf.save(path)
                uploaded_files.append(filename)

                process_pdf(path, pdf_id)
                global_state[pdf_id] = {'filename': filename}

            session['chat_history'] = []
            session['uploaded_files'] = uploaded_files
            session['pdf_ids'] = list(global_state.keys())
            return redirect(url_for('index'))

        elif 'question' in request.form:
            question = request.form['question']
            pdf_ids = session.get('pdf_ids', [])

            # 🧠 Create a full context from chat history (limit to last 3 turns for performance)
            context_turns = chat_history[-3:] if len(chat_history) > 3 else chat_history
            context_prompt = "\n".join(
                f"User: {turn['question']}\nAssistant: {turn['answer']}" for turn in context_turns
            )
            full_prompt = f"{context_prompt}\nUser: {question}"

            compare_match = re.findall(r'(paper_\d+)', question.lower())
            if len(compare_match) >= 2:
                paper1, paper2 = compare_match[:2]
                chunks1 = retrieve_chunks_chroma(question, paper1)
                chunks2 = retrieve_chunks_chroma(question, paper2)

                answer1 = generate_answer(f"{full_prompt} (from {paper1})", chunks1)
                answer2 = generate_answer(f"{full_prompt} (from {paper2})", chunks2)

                answer = f"📄 **{global_state[paper1]['filename']}**:\n{answer1}\n\n📄 **{global_state[paper2]['filename']}**:\n{answer2}"
            else:
                all_chunks = []
                for pdf_id in pdf_ids:
                    chunks = retrieve_chunks_chroma(question, pdf_id)
                    all_chunks.extend(chunks)

                answer = generate_answer(full_prompt, all_chunks)

            chat_history.append({'question': question, 'answer': answer})
            session['chat_history'] = chat_history

    return render_template('index.html', chat_history=chat_history, uploaded=uploaded,
                           uploaded_count=len(uploaded_files), pdf_ids=pdf_ids, uploaded_files=uploaded_files)
@app.route('/reset', methods=['POST'])
def reset():
    session.clear()
    global_state.clear()
    chroma_client.reset()
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)