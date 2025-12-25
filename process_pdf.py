from pathlib import Path
from pypdf import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

def process_pdf(pdf_path_str, disease_db):
    # Path setup - Ensure this is a string for Chroma
    current_dir = Path.cwd()
    DB_DIR = current_dir / disease_db
    pdf_path = Path(pdf_path_str)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"Cannot find the file at: {pdf_path}")

    # 2. Extract Text
    reader = PdfReader(str(pdf_path))
    full_text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            full_text += content

    if not full_text.strip():
        raise ValueError("The PDF appears to be empty or contains only images (OCR required).")

    # 3. Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(full_text)
    
    # 4. Vectorization
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # FIX: Convert DB_DIR to a string using str()
    # Also use .as_posix() inside str() to ensure forward slashes
    vectorstore = Chroma.from_texts(
        texts=chunks, 
        embedding=embeddings,
        persist_directory=str(DB_DIR.as_posix()) 
    )
    return vectorstore
