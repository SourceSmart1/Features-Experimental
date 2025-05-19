import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

# Page config must be the first Streamlit command
st.set_page_config(page_title="Model: Llama4-Maverick", layout="wide")

# Load environment variables
load_dotenv()
api_token = os.getenv("LLM_API_KEY")

# Initialize OpenAI client
client = OpenAI(
    api_key=api_token,
    base_url="https://api.llmapi.com/"
)

# Model configurations
MODELS = {
    # Llama Models
    "llama4-maverick": {"params": "70B", "price": 0.009, "description": "Maverick version of Llama 4, optimized for creative tasks and complex reasoning."},
    "llama4-scout": {"params": "70B", "price": 0.009, "description": "Scout version of Llama 4, specialized in fast and adaptive responses."},
    "llama3.3-70b": {"params": "70B", "price": 0.0028, "description": "Llama 3.3 update with 70 billion parameters, enhanced for text generation."},
    "llama3.1-405b": {"params": "405B", "price": 0.003596, "description": "Large-scale Llama 3.1 with 405 billion parameters for high-demand applications."},
    
    # DeepSeek Models
    "deepseek-r1": {"params": "70B", "price": 0.009, "description": "DeepSeek R1, first generation focused on reasoning and information retrieval."},
    "deepseek-v3": {"params": "70B", "price": 0.0028, "description": "DeepSeek V3, third generation with improvements in comprehension."},
    
    # Mistral Models
    "mistral-7b-instruct": {"params": "7B", "price": 0.0004, "description": "Mistral model with 7 billion parameters, fine-tuned for instruction and assistance tasks."},
    "mixtral-8x7b-instruct": {"params": "56B (8x7B)", "price": 0.0028, "description": "Mixtral with 8 experts of 7B each, optimized for chat and instruction-driven tasks."},
    "mixtral-8x22b-instruct": {"params": "176B (8x22B)", "price": 0.0028, "description": "Mixtral with 8 experts of 22B each, fine-tuned for precise instruction following."},
    "Nous-Hermes-2-Mixtral-8x7B-DPO": {"params": "56B (8x7B)", "price": 0.0004, "description": "MoE combined model, fine-tuned with DPO, using 8 experts of 7B each for high performance in language tasks."},
    
    # Code Models
    "coder-large": {"params": "32B", "price": 0.0016, "description": "Coder-Large with 32 billion parameters, fine-tuned for code generation."},
    "Qwen2.5-Coder-32B-Instruct": {"params": "32B", "price": 0.0016, "description": "Specialized in code generation and debugging."},
    
    # Other Models
    "virtuoso-large": {"params": "72B", "price": 0.0019, "description": "Virtuoso-Large with 72 billion parameters, Arcee AI's flagship model."},
    "maestro-reasoning": {"params": "32B", "price": 0.0042, "description": "Maestro-Reasoning with 32 billion parameters, specialized in advanced reasoning."},
}

# Custom CSS for better styling
st.markdown("""
    <style>
    .model-param {
        color: #1f77b4;
        font-weight: 500;
    }
    .model-price {
        color: #2ca02c;
        font-weight: 500;
    }
    .model-desc {
        color: #666666;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for model selection
with st.sidebar:
    st.title("Model Selection")
    
    # Model selection
    model = st.selectbox("Select Model", list(MODELS.keys()))
    
    # Display model details with custom styling
    st.markdown("### Model Details")
    st.markdown(f"""
        <p><span class="model-param">Parameters:</span> {MODELS[model]['params']}</p>
        <p><span class="model-price">Price per token:</span> ${MODELS[model]['price']}</p>
        <p><span class="model-desc">{MODELS[model]['description']}</span></p>
    """, unsafe_allow_html=True)

# Main chat interface
st.title(f"Procurement Expert: {model}")

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
    
    # Get AI response with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        chat_completion = client.chat.completions.create(
            messages=st.session_state.messages,
            model=model,
            stream=True
        )
        
        for chunk in chat_completion:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")
        
        # Final update without the cursor
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
