# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 15:12:40 2025

@author: LENOVO
"""

from sentence_transformers import SentenceTransformer
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
import os

PDF_FOLDER = "./Sample_datasets"
VECTOR_DB_PATH = "./chroma_db"
EMBED_MODEL_PATH = "./all-MiniLM-L6-v2-local"

embedding_model = SentenceTransformer(EMBED_MODEL_PATH)
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = chroma_client.get_or_create_collection(name="documents")

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def index_pdfs():
    collection.delete()  # Clear previous index if needed
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            text = extract_text_from_pdf(pdf_path)
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                embedding = embedding_model.encode(chunk)
                collection.add(
                    documents=[chunk],
                    metadatas=[{"source": filename}],
                    ids=[f"{filename}_{i}"]
                )
            print(f"Indexed: {filename}")
    print("All PDFs indexed and stored in Chroma DB.")

if __name__ == "__main__":
    index_pdfs()
