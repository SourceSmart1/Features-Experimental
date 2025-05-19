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

    # System Prompt Editor
    with st.expander("System Prompt", expanded=False):
        st.markdown("""
            <p><span class="model-param">System Prompt</span> defines the AI's behavior and role. 
            This sets the context for how the model should respond to your queries.</p>
        """, unsafe_allow_html=True)
        system_prompt = st.text_area(
            "Edit System Prompt",
            value="""You are an elite procurement and supply chain expert with decades of experience across multiple industries. Your expertise includes:

1. Strategic Sourcing & Supplier Management
- Supplier evaluation and selection
- Contract negotiation and management
- Risk assessment and mitigation
- Cost optimization strategies
- Supplier relationship management

2. Procurement Best Practices
- Category management
- Spend analysis
- Procurement process optimization
- Digital procurement solutions
- Sustainable procurement practices

3. Market Intelligence
- Industry trends and insights
- Market analysis and forecasting
- Price benchmarking
- Supply market dynamics
- Global sourcing strategies

4. Compliance & Risk Management
- Regulatory compliance
- Ethical sourcing
- Quality assurance
- Supply chain security
- Business continuity planning

When responding to queries:
- Provide detailed, actionable insights based on industry best practices
- Include relevant examples and case studies when applicable
- Consider both short-term and long-term implications
- Address cost, quality, and risk factors
- Suggest practical implementation steps
- Reference current market conditions and trends
- Highlight potential challenges and mitigation strategies

Your goal is to help users make informed procurement decisions that drive value, reduce risk, and create sustainable competitive advantages for their organizations.""",
            height=300
        )

    # API Parameters under expandable section
    with st.expander("Advanced Parameters", expanded=False):
        # Temperature
        st.markdown("""
            <p><span class="model-param">Temperature</span> controls randomness in the output. 
            Lower values (like 0.2) make responses more focused and deterministic, 
            while higher values (like 0.8) make responses more creative and diverse.</p>
        """, unsafe_allow_html=True)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        
        # Max Tokens
        st.markdown("""
            <p><span class="model-param">Max Tokens</span> limits the length of the response. 
            One token is roughly 4 characters or 3/4 of a word. 
            Higher values allow for longer responses but may increase costs.</p>
        """, unsafe_allow_html=True)
        max_tokens = st.slider("Max Tokens", min_value=100, max_value=4000, value=2000, step=100)
        
        # Top P
        st.markdown("""
            <p><span class="model-param">Top P</span> (nucleus sampling) controls diversity via probability mass. 
            Lower values (like 0.1) make responses more focused, 
            while higher values (like 0.9) allow for more diverse outputs.</p>
        """, unsafe_allow_html=True)
        top_p = st.slider("Top P", min_value=0.1, max_value=1.0, value=0.9, step=0.1)
        
        # Frequency Penalty
        st.markdown("""
            <p><span class="model-param">Frequency Penalty</span> reduces repetition of the same line verbatim. 
            Higher values (like 1.0) make the model less likely to repeat the same line.</p>
        """, unsafe_allow_html=True)
        frequency_penalty = st.slider("Frequency Penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
        
        # Presence Penalty
        st.markdown("""
            <p><span class="model-param">Presence Penalty</span> reduces repetition of the same topic. 
            Higher values (like 1.0) make the model more likely to talk about new topics.</p>
        """, unsafe_allow_html=True)
        presence_penalty = st.slider("Presence Penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)

# Main chat interface
st.title(f"Procurement Expert: {model}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]
else:
    # Update system message if it's changed
    if st.session_state.messages[0]["content"] != system_prompt:
        st.session_state.messages[0]["content"] = system_prompt

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
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        
        for chunk in chat_completion:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")
        
        # Final update without the cursor
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
