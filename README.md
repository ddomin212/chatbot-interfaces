## Chatbot interfaces
I was trying to build a chatbot with a streaming interface, but I couldn't find any good examples. So I decided to build one myself. I hope this helps someone else.

`cl_app.py` uses the **chainlit** library, it does stream the response but its nested inside the input box and i could nto find how to actually make it stream the tokens.
`st_app.py` uses the **streamlit** library, it does stream the response, using the `StreamingResponseHandler`, it literally took me a few hours to find out how to do it, so I hope this helps someone else.