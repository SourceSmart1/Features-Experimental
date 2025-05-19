import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_token = os.getenv("LLM_API_KEY")

# Initialize OpenAI client
client = OpenAI(
    api_key=api_token,
    base_url="https://api.llmapi.com/"
)

# Page config
st.set_page_config(page_title="AI Chat Assistant", layout="wide")
st.title("AI Chat Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are a the best procurement expert in the world. You are given a question and you need to answer it based on your knowledge and experience."
        }
    ]

# Display chat messages from history
for message in st.session_state.messages:
    if message["role"] != "system":  # Don't display system messages
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like help with?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chat_completion = client.chat.completions.create(
                messages=st.session_state.messages,
                model="llama4-maverick",
                stream=False
            )
            response = chat_completion.choices[0].message.content
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
