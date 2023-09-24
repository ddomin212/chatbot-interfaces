from langchain import LLMChain
from langchain.llms import CTransformers
import chainlit as cl
import os
import tempfile
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers, Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from io import BytesIO
from chainlit.input_widget import Select, Slider

@cl.on_chat_start
async def main():
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="LLama2 Model",
                values=["llama2-7b"],
                initial_index=0,
            ),
            Slider(
                id="Temperature",
                label="OpenAI - Temperature",
                initial=0.1,
                min=0,
                max=1,
                step=0.05,
            )]).send()
    
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!", accept=["text/plain", "application/pdf", "application/msword"], max_size_mb=10, max_files=5
        ).send()
    
    await cl.Message(
        content=f"Processing content of {len(files)} files..."
    ).send()

    if files:
        text = []
        for ask_file in files:
            file = BytesIO(ask_file.content)
            file_extension = os.path.splitext(ask_file.name)[1]
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
                await cl.Message(
                    content=f"File type {file_extension} not supported!"
                ).send()

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

        llm = CTransformers(model="../llama-2-7b-chat.ggmlv3.q4_0.bin",
                            streaming=True, 
                            callbacks=[StreamingStdOutCallbackHandler()],
                            model_type="llama", config={'max_new_tokens': 500, 'temperature': settings["Temperature"]})
        
        # llm = Replicate(
        #     streaming = True,
        #     model = "replicate/llama-2-13b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781", 
        #     callbacks=[StreamingStdOutCallbackHandler()],
        #     model_kwargs = {"temperature": 0.9, "max_length": 512,"top_p": 1})

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        llm_chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                    memory=memory)

        cl.user_session.set("llm_chain", llm_chain)

        await cl.Message(
            content=f"All is ready, you can start chatting!"
        ).send()



@cl.on_message
async def main(message: str):
        # Retrieve the chain from the user session
        llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

        # Call the chain asynchronously
        res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True)])

        # Do any post processing here

        # "res" is a Dict. For this chain, we get the response by reading the "text" key.
        # This varies from chain to chain, you should check which key to read.
        msg = cl.Message(content="")
        for token in res["answer"]:
            await msg.stream_token(token)

        await msg.send()

@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)