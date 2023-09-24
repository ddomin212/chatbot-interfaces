import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler

class StreamingResponseHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self.full_response = ''

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.full_response += token
        st.session_state.messages_field.markdown(self.full_response)