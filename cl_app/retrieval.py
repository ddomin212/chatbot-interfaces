import os
import tempfile
from io import BytesIO

from chainlit import AskFileMessage
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader


def load_uploaded_file(up_file: AskFileMessage, text: list[str]) -> bool:
    file = BytesIO(up_file.content)
    file_extension = os.path.splitext(up_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    loader = None
    if file_extension == ".pdf":
        loader = PyPDFLoader(temp_file_path)
    elif file_extension == ".docx" or file_extension == ".doc":
        loader = Docx2txtLoader(temp_file_path)
    elif file_extension == ".txt":
        loader = TextLoader(temp_file_path)
    else:
        return False

    if loader:
        text.extend(loader.load())
        os.remove(temp_file_path)
        return True


def create_vector_db(text: list[str]) -> FAISS:
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                        model_kwargs={'device': 'cpu'})

    # Create vector store
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    return vector_store