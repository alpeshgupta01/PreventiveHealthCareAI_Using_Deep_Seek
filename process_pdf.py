import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def process_pdf(file_path, db_name):
    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Split text
    # DeepSeek has a 128k context window, so we can use larger chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)

    # 3. Use BGE Embeddings (A modern replacement for all-MiniLM)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. Create Vector DB
    vector_db = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings, 
        persist_directory=f"./{db_name}"
    )
    return vector_db