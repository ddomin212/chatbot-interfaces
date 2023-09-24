import chainlit as cl
from chainlit.input_widget import Slider
from retrieval import load_uploaded_file, create_vector_db
from chatbot import load_chatbot

@cl.on_chat_start
async def main() -> None:
    # define the settings for the chatbot
    settings = await cl.ChatSettings(
        [
            Slider(
                id="Temperature",
                label="Temperature",
                initial=0.1,
                min=0,
                max=1,
                step=0.05,
            )
        ]
    ).send()
    
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!", accept=["text/plain", "application/pdf", "application/msword"], max_size_mb=10, max_files=5
        ).send()

    # start parsing the uploaded files
    if files:
        text = []
        for up_file in files:
            good = load_uploaded_file(up_file, text)
            if good == False:
                await cl.Message(
                    content=f"File {up_file.name} is not supported, please upload a text file to begin!"
                ).send()
                return

        # create a vector database from the parsed text
        vector_store = create_vector_db(text)

        # load the chatbot chain
        llm_chain = load_chatbot(vector_store, settings)

        # save the chain to the user session (async fun stuff)
        cl.user_session.set("llm_chain", llm_chain)

        # send message to the user that everything is ready
        await cl.Message(
            content=f"All is ready, you can start chatting!"
        ).send()



@cl.on_message
async def main(message: str) -> None:
        llm_chain = cl.user_session.get("llm_chain")

        res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True)])

        msg = cl.Message(content="")
        for token in res["answer"]:
            await msg.stream_token(token)

        await msg.send()

@cl.on_settings_update
async def setup_agent(settings) -> None:
    print("on_settings_update", settings)