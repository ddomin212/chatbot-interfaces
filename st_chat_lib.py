import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()


def initialize_session_state():
    """Initializes the session state variables for chatting. These include the chat history, the generated responses, and the past user inputs."""
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    """Runs the chain, gets the generated response and saves it as a tulpe with users query in chat history.

    Args:
        query: user input
        chain: LangChain chain
        history: history of messages

    Returns:
        answer
    """
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    """displays the chat history and the input field for the user to ask questions.

    Args:
        chain: LangChain chain, which is used to generate response (essentialy a prompt template with variable input)
    """
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def create_conversational_chain(vector_store):
    """Creates a chatbot chain with memory

    Args:
        vector_store: vector db (FAISS)

    Returns:
        chatbot chain
    """
    llm = CTransformers(model="../llama-2-7b-chat.ggmlv3.q4_0.bin",
                        streaming=True, 
                        callbacks=[StreamingStdOutCallbackHandler()],
                        model_type="llama", config={'max_new_tokens': 500, 'temperature': 0.01})
    
    # llm = Replicate(
    #     streaming = True,
    #     model = "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781", 
    #     callbacks=[StreamingStdOutCallbackHandler()],
    #     model_kwargs = {"temperature": 0.9, "max_length": 512,"top_p": 1})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain

def main():
    initialize_session_state()
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
    st.title("RAG Chatbot ❓ 🦜")
    st.sidebar.title("Upload your company documents")
    uploaded_files = st.sidebar.file_uploader("Supported formats include: .pdf, .docx, .doc, .txt, .csv", accept_multiple_files=True)


    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
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
            # elif file_extension == ".csv":
            #     with st.sidebar:
            #         st.write("**CSV file detected.** Please specify delimiter")
            #         delimiter = st.text_input("Delimiter", value=",", max_chars=1)
            #         submit_button = st.button(label='Submit')
            #         if submit_button and delimiter and len(delimiter) == 1:
            #             if delimiter not in [",", ";", "\t"]:
            #                 st.error("Delimiter not supported")
            #                 continue
            #             else:
            #                 loader = CSVLoader(file_path=temp_file_path, encoding="utf-8", csv_args={'delimiter': delimiter})
            else:
                st.error("File type not supported")
                continue

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
        text_chunks = text_splitter.split_documents(text)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                           model_kwargs={'device': 'cpu'})

        # Create vector store
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        # Create the chain object
        chain = create_conversational_chain(vector_store)

        
        display_chat_history(chain)

if __name__ == "__main__":
    main()
