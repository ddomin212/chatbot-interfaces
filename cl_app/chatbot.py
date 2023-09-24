from langchain.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import FAISS

def load_chatbot(vector_store: FAISS, settings: dict[str, float|int|str|bool]) -> ConversationalRetrievalChain:
    llm = CTransformers(model="../llama-2-7b-chat.ggmlv3.q4_0.bin",
                            streaming=True, 
                            callbacks=[StreamingStdOutCallbackHandler()],
                            model_type="llama", config={'max_new_tokens': 500, 'temperature': settings["Temperature"]})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    llm_chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                memory=memory)
    
    return llm_chain