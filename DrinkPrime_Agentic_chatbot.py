# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 02:04:20 2025

@author: LENOVO
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
import google.generativeai as genai
import os 
import os
import chromadb
import numpy as np
import pandas as pd
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch
import pdfplumber
import pytesseract
import cv2
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Configure Gemini API key (make sure to set it securely)
#genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
genai.configure(api_key="AIzaSyDrJgGVayIdX5qxQBcef0wtIIcmmS_YZ_w")  # Your key from AI Studio
gemini_model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")

PDF_FOLDER = os.environ.get("PDF_FOLDER", "./Sample_datasets")
VECTOR_DB_PATH = os.environ.get("VECTOR_DB_PATH", "./chroma_db")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = chroma_client.get_or_create_collection(name="documents")

def extract_text_from_pdf(pdf_path):
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            #  Extract Text
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

            #  Extract Tables
            tables = page.extract_table()
            if tables:
                for table in tables:
                    text += "\n".join([" | ".join(row) for row in table if row]) + "\n"

            #  Extract Text from Images
            for img in page.images:
                image_data = page.to_image()
                image = image_data.original
                image_array = np.array(image)
                image_gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
                extracted_text = pytesseract.image_to_string(Image.fromarray(image_gray))
                if extracted_text:
                    text += "\n[Image Text Extracted]: " + extracted_text + "\n"

    return text.strip()
def add_document_with_check(document_text, doc_id, metadata=None):
    existing_docs = collection.get(ids=[str(doc_id)])  # Check if ID exists
    if len(existing_docs["ids"]) == 0:  # If ID does not exist, add it
        collection.add(
            documents=[document_text],
            metadatas=[metadata] if metadata else [{}],
            ids=[str(doc_id)]
        )
        #print(f" Added new document with ID: {doc_id}")
    else:
        print(f"⚠️ Skipping duplicate document with ID: {doc_id}")

def process_pdfs():
    documents = []
    doc_id = 0  # Initialize a document ID counter
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            #print(f"📖 Processing: {filename}")
            text = extract_text_from_pdf(pdf_path)

            #  Split Text into Chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(text)

            #  Store Chunks in ChromaDB
            for chunk in chunks:
                #embedding = embedding_model.encode(chunk).tolist()
                # Use the doc_id counter for unique IDs and increment it
                #collection.add(documents=[chunk], metadatas=[{"source": filename}], ids=[str(doc_id)])
                #doc_id += 1
                add_document_with_check(chunk, str(collection.count()), {"source": filename})
                documents.append(chunk)

    return documents

#  BM25 Keyword Search
def bm25_search(query, documents):
    tokenized_corpus = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    ranked_docs = [documents[i] for i in np.argsort(scores)[::-1][:3]]
    return ranked_docs

#  ChromaDB Vector Search
def vector_search(query):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=1)
    return [doc for doc in results["documents"][0]]


def generate_response(query, retrieved_docs, chat_history=[]):
    """
    Generate response using Gemini, including conversational history for context.

    Parameters:
    - query: latest user input
    - retrieved_docs: top retrieved context chunks
    - chat_history: list of previous turns [{"user": "...", "ai": "..."}]
    """

    # Handle greetings simply
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
    if query.lower().strip() in greetings:
        return "Hello! 👋 How can I assist you today?"

    # Prepare past conversation
    formatted_history = ""
    if chat_history:
        formatted_history = "\n".join([f"User: {turn['user']}\nAI: {turn['ai']}" for turn in chat_history])

    # Context from retrieval
    context = "\n".join(retrieved_docs[:2])

    # Prompt for Gemini
    prompt = f"""
    You are a friendly, empathetic customer support agent for DrinkPrime.
    When the user asks a question, respond:

    - In a warm and helpful tone
    - With step-by-step guidance if needed
    - Using emojis to keep it human and engaging
    - Always end by asking if they need more help

    📘 Knowledge Context:
    {context}

    💬 Conversation History:
    {formatted_history}

    🧑‍💻 Latest User Question:
    {query}

    🤖 Your helpful and engaging answer:
    """

    # Call Gemini model
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

docs = []
docs_loaded = False

@app.before_request
def load_documents():
    global docs, docs_loaded
    if not docs_loaded:
        print("🔁 Initializing RAG system...")
        docs = process_pdfs()
        docs_loaded = True
        print(f"✅ Loaded {len(docs)} chunks from PDFs")

@app.route("/rag-query", methods=["POST"])
def rag_query():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    keyword_results = bm25_search(query, docs)
    vector_results = vector_search(query)
    all_chunks = keyword_results + vector_results

    answer = generate_response(query, all_chunks)

    return jsonify({
        "query": query,
        "answer": answer
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080) #this part is acting like api