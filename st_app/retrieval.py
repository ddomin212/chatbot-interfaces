import os
import tempfile
from io import BytesIO

import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader


def save_as_temp_file(file: BytesIO) -> str:
    """Saves the uploaded file as a temporary file on disk, because of langchin document loaders
    
    Args:
        file: uploaded file
    """
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name
    return temp_file_path

def load_uploaded_file(file: BytesIO, temp_file_path: str) -> PyPDFLoader | Docx2txtLoader | TextLoader | None:
    """Loads the uploaded file with the appropriate document loader
    
    Args:
        file: uploaded file
        temp_file_path: path to the temporary uploaded file on disk
    """
    file_extension = os.path.splitext(file.name)[1]

    loader = None
    if file_extension == ".pdf":
        loader = PyPDFLoader(temp_file_path)
    elif file_extension == ".docx" or file_extension == ".doc":
        loader = Docx2txtLoader(temp_file_path)
    elif file_extension == ".txt":
        loader = TextLoader(temp_file_path)
    else:
        st.error("File type not supported")
    
    return loader

def parse_uploaded_document(file: BytesIO, text: list[str]) -> None:
    """Parses the uploaded file and appends the corpus of text (knowledge base)
    
    Args:
        file: uploaded file
        text: list of strings, which is used as a corpus of text
    """
    
    temp_file_path = save_as_temp_file(file)

    loader = load_uploaded_file(file, temp_file_path)

    if loader:
        text.extend(loader.load())
        os.remove(temp_file_path)

def create_vector_db(text: list[str]) -> FAISS:
    """Creates a vector store from the corpus of text
    
    Args:
        text: list of strings, which is used as a corpus of text
    
    Returns:
        FAISS vector store (database)
    """
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=40)
    text_chunks = text_splitter.split_documents(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                        model_kwargs={'device': 'cpu'})

    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    return vector_store
    