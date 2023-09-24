import streamlit as st
from dotenv import load_dotenv

from retrieval import parse_uploaded_document, create_vector_db
from setup import init_st_session_state, create_conversational_chain, render_st_elements
from chat import display_chat_history

load_dotenv()

def main():
    init_st_session_state()
    temp_slider, uploaded_files = render_st_elements()


    if uploaded_files:
        text = []
        for file in uploaded_files:
            parse_uploaded_document(file, text)
        vector_store = create_vector_db(text)
        chain = create_conversational_chain(vector_store, temp_slider)
        display_chat_history(chain)

if __name__ == "__main__":
    main()
