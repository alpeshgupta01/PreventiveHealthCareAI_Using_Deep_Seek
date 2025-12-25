import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # Change this
from langchain_community.vectorstores import Chroma

def process_pdf(file_path, db_name):
    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # 3. Use Hugging Face Embeddings (No OpenAI key needed here)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Create Vector DB
    vector_db = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings, 
        persist_directory=f"./{db_name}"
    )
    return vector_db