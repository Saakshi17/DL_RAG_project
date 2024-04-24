# app.py

import streamlit as st
from setup import setup_wikipedia, setup_arxiv, setup_vector_store, setup_openai_model, setup_openai_tools, invoke_query_noRAG

# Set up components
wiki_query_runner = setup_wikipedia()
arxiv_query_runner = setup_arxiv()
vectordb = setup_vector_store()
llm=setup_openai_model()
# Create Streamlit app
st.title('LangChain Wikipedia Assistant')


prompt = st.text_input("Enter your question or prompt:")
print(prompt)

if st.button('Generate Answer'):
    if prompt:
        response=invoke_query_noRAG(prompt)
        st.write("Answer:", response)
    else:
        st.write("Please enter a prompt.")
