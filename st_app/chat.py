import streamlit as st

from langchain.chains import ConversationalRetrievalChain

def get_chatbot_response(query: str, chain: ConversationalRetrievalChain, history: list[dict[str, str]]) -> str:
    """Runs the chain, gets the generated response and saves it as a tulpe with users query in chat history.

    Args:
        query: user input
        chain: LangChain chain
        history: history of messages

    Returns:
        answer
    """
    result = chain({"question": query, "chat_history": history})
    return result["answer"]

def process_user_input(prompt: str) -> None:
    """Adds the user input to the chat history.

    Args:
        prompt: user input
    """
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

def process_chatbot_response(prompt: str, chain: ConversationalRetrievalChain) -> None:
    """Generates the chatbot response and adds it to the chat history.

    Args:
        prompt: user input
        chain: LangChain chain
    """
    with st.chat_message("assistant"):
        placeholder = st.empty()
        st.session_state.messages_field = placeholder
        response = get_chatbot_response(prompt, chain, st.session_state.messages)
        placeholder.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

def display_chat_history(chain: ConversationalRetrievalChain) -> None:
    """Displays the chat history and the input field for the user to ask questions.

    Args:
        chain: LangChain chain, which is used to generate response (essentialy a prompt template with variable input)
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        process_user_input(prompt)
        with st.spinner("Generating response..."):
            process_chatbot_response(prompt, chain)