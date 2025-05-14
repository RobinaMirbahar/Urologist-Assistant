import streamlit as st
import google.generativeai as genai
import os
import time
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.warning("python-dotenv not installed. Using environment variables directly")

# UI Setup
st.set_page_config(
    page_title="Urology Medical Assistant", 
    page_icon="⚕️",
    layout="wide"
)

st.title("⚕️ Urology Medical Assistant (Gemini 2.5 Pro)")
st.markdown("Get information about urological conditions with symptoms and image references")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "🔑 Enter your Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", "")
    )
    
    st.session_state.user_type = st.radio(
        "Select your role:",
        ("Patient", "Medical Student", "Practicing Urologist"),
        index=0
    )
    
    st.markdown("### Model Settings")
    temperature = st.slider("Response specificity", 0.0, 1.0, 0.3)
    max_tokens = st.slider("Max response length", 512, 8192, 2048, step=512)
    
    st.markdown("---")
    if st.button("🧹 Clear Chat History"):
        st.session_state.clear()
        st.rerun()

# System Instruction
system_instruction = """You are an expert urology assistant that provides accurate medical information about urological conditions. 
Adapt your responses based on the user's role:

For Patients:
- Use simple, non-technical language
- Focus on symptoms, basic explanations, and when to seek care
- Provide reassurance without diagnosis
- Include basic prevention tips

For Medical Students:
- Provide detailed anatomical and physiological explanations
- Include differential diagnoses
- Explain diagnostic pathways
- Reference relevant studies and guidelines

For Practicing Urologists:
- Provide latest treatment guidelines
- Include surgical considerations when appropriate
- Reference recent studies and meta-analyses
- Discuss complex cases and comorbidities

For all responses:
1. Start with a clear definition/description of the condition
2. List key symptoms (with bold headings)
3. Include typical diagnostic methods
4. Mention treatment options (tailored to user type)
5. When appropriate, describe what visual findings might look like (imaging, cystoscopy, etc.)
6. Always include disclaimer that this is not medical advice
"""

# Initialize the model
if api_key:
    genai.configure(api_key=api_key)
    
    if "model" not in st.session_state:
        st.session_state.model = genai.GenerativeModel(
            model_name="gemini-2.5-pro-preview-03-25",
            generation_config={
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 50,
                "max_output_tokens": max_tokens,
            },
            system_instruction=system_instruction
        )
    
    if "chat" not in st.session_state:
        st.session_state.chat = st.session_state.model.start_chat(history=[])

# Chat Interface
if api_key:
    # Display chat history
    for message in st.session_state.chat.history:
        if message.role == "user":
            with st.chat_message("user"):
                st.markdown(f"**You:** {message.parts[0].text}")
        else:
            with st.chat_message("assistant"):
                st.markdown(message.parts[0].text)

    # Handle user input
    if prompt := st.chat_input(f"Ask about a urological condition as a {st.session_state.user_type}..."):
        st.session_state.last_question = prompt
        with st.chat_message("user"):
            st.markdown(f"**You:** {prompt}")
        
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                start_time = time.time()
                try:
                    full_prompt = f"""User type: {st.session_state.user_type}
                    
                    Query: {prompt}
                    
                    Please provide:
                    1. Level-appropriate explanation
                    2. Key symptoms with bold headings
                    3. Diagnostic approaches
                    4. Treatment overview
                    5. Visual findings description
                    6. Appropriate disclaimers"""
                    
                    response = st.session_state.chat.send_message(
                        full_prompt,
                        stream=True
                    )
                    
                    response_text = ""
                    container = st.empty()
                    for chunk in response:
                        response_text += chunk.text
                        container.markdown(response_text + "▌")
                    
                    container.markdown(response_text)
                    response_time = time.time() - start_time
                    
                    # Estimate token count (rough approximation)
                    token_count = len(response_text.split())
                    st.caption(f"Generated in {response_time:.2f}s | ~{token_count} tokens | Max: {max_tokens}")
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
else:
    st.info("Please enter your Gemini API key in the sidebar to begin")

# About Section
with st.expander("ℹ️ About This Assistant"):
    st.markdown("""
    **Urology Assistant Features:**
    - Powered by Gemini 2.5 Pro (preview-03-25)
    - Tailored responses for patients, students, and professionals
    - Symptom lists with visual findings descriptions
    - Diagnostic and treatment information
    
    **Disclaimer:**
    This tool provides general medical information only and is not a substitute 
    for professional medical advice, diagnosis, or treatment.
    
    **Requirements:**
    ```bash
    streamlit>=1.32
    google-generativeai>=0.3
    python-dotenv>=1.0
    ```
    """)
