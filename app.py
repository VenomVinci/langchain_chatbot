from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# Get OpenAI API key from .env
openai_api_key = os.getenv("api_key")

# Error if the key is missing.
if not openai_api_key:
    st.error(" Missing OpenAI API Key. Please check your .env file.")
    st.stop()

# Set up the OpenAI LLM via LangChain
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

# Define system and user prompt structure
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])

chain = prompt | llm | StrOutputParser()


st.set_page_config(page_title="LangChain Chatbot", page_icon="🤖")
st.title(" AI Assistant (LangChain + OpenAI)")
st.markdown("Ask me anything:")

user_input = st.text_input("Your question:")

if user_input:
    with st.spinner("Thinking..."):
        try:
            response = chain.invoke({"input": user_input})
            st.success(response)
        except Exception as e:
            st.error(f" Error: {str(e)}")




