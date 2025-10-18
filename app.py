import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import time
import re
import html
import textwrap

# NLP Libraries (assuming these are installed)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Configuration ---
st.set_page_config(
    page_title="Mental Health Guardian",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Advanced "Aurora" Dark Theme & Custom CSS ---
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background-image: linear-gradient(180deg, #1f2937, #111827);
        color: #e5e7eb;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: rgba(17, 24, 39, 0.8);
        border-right: 1px solid #4b5563;
    }

    /* Chat input box */
    .stTextInput > div > div > input {
        background-color: #374151;
        color: #e5e7eb;
        border: 1px solid #4b5563;
        border-radius: 0.5rem;
    }

    /* General text and headers */
    h1, h2, h3 {
        color: #60a5fa;
    }

    /* Chat message container */
    .chat-container {
        margin-bottom: 1rem;
    }

    /* Chat bubble base style */
    .chat-bubble {
        padding: 1.2rem;
        border-radius: 0.75rem;
        margin-bottom: 0.5rem;
        border: 1px solid transparent;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .chat-bubble:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
    }

    /* User chat bubble */
    .chat-bubble-user {
        background: linear-gradient(to right, #3b82f6, #60a5fa);
        color: #ffffff;
        margin-left: auto;
        width: fit-content;
        max-width: 80%;
    }

    /* Bot chat bubble */
    .chat-bubble-bot {
        background: #374151;
        color: #e5e7eb;
        margin-right: auto;
        width: fit-content;
        max-width: 90%;
    }

    .message-time {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-top: 0.3rem;
        text-align: right;
    }

    .confidence-score {
        font-size: 0.8rem;
        color: #d1d5db;
        font-style: italic;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid #4b5563;
    }

    /* Alert box in the sidebar */
    .stAlert {
        background-color: rgba(251, 191, 36, 0.1);
        border: 1px solid #fbbf24;
        border-radius: 0.5rem;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #3b82f6;
        color: #fff;
        border-radius: 0.5rem;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# --- Caching and Model Loading ---
@st.cache_resource
def load_models():
    """Loads the sentence transformer model and pre-computed embeddings."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        question_embeddings = np.load('question_embeddings.npy')
        return model, question_embeddings
    except FileNotFoundError:
        st.error("Critical Error: question_embeddings.npy not found. Please generate embeddings first.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading models: {e}")
        return None, None

@st.cache_data
def load_data():
    """Loads the FAQ dataset."""
    try:
        df = pd.read_csv('cleaned_mental_health_faq.csv')
        return df
    except FileNotFoundError:
        st.error("Critical Error: cleaned_mental_health_faq.csv not found. Make sure the file is in the correct directory.")
        return None

# --- Core Chatbot Logic ---
def get_high_level_response(user_query, df, model, question_embeddings, threshold=0.5):
    """Generates an empathetic and detailed response."""
    if not user_query or user_query.isspace():
        return "Please feel free to ask me anything about mental health.", 0.0, None

    normalized_query = user_query.lower().strip()

    # --- Handle Greetings and Farewells ---
    greetings = ['hi', 'hello', 'hey', 'yo', 'greetings']
    farewells = ['bye', 'goodbye', 'see you', 'farewell', 'take care']

    if normalized_query in greetings:
        response = "Hello there! I'm here to help with any questions you have about mental health. What's on your mind?"
        return response, 1.0, "Greeting"
    
    if normalized_query in farewells:
        response = "Goodbye! Remember to take care of yourself. I'm here if you need me again."
        return response, 1.0, "Farewell"
    
    # --- Existing Logic for FAQ search ---
    query_embedding = model.encode([user_query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, question_embeddings).flatten()
    best_match_idx = similarities.argmax()
    best_similarity = similarities[best_match_idx]

    if best_similarity >= threshold:
        matched_question = df['Questions'].iloc[best_match_idx]
        answer = df['Answers'].iloc[best_match_idx]
        answer = re.sub(r'\b([A-Z]{2,})\b', r'**\1**', answer)
        intro = f"That's a great question about **'{matched_question.strip()}'**. Here's some information that might help:\n\n"
        outro = "\n\nI hope this explanation is helpful. Remember, this is for informational purposes only. For personal advice, please consult a healthcare professional."
        response = intro + answer + outro
        return response, best_similarity, matched_question
    else:
        response = textwrap.dedent("""
            I'm sorry, but I couldn't find a direct answer to your question in my knowledge base.

            Could you try rephrasing it? Sometimes, asking in a different way can help me understand better.

            **For example, instead of:**
            * "how to prevent from mental health"

            **You could try:**
            * "how can I protect my mental well-being?" or "what are strategies for good mental health?"

            If you need immediate support, please reach out to a mental health professional or a crisis hotline. Your well-being is the top priority.
        """)
        return response, 0.0, None

# --- UI Rendering Functions ---
def display_chat_message(role, message, timestamp, confidence=None, matched_q=None):
    """Displays a single chat message with proper markdown rendering."""
    col1, col2 = st.columns([1, 10])
    
    if role == "user":
        with col2:
            with st.container(border=False):
                st.markdown(f'<div class="chat-bubble chat-bubble-user"><strong>You:</strong></div>', unsafe_allow_html=True)
                st.markdown(message)
                st.markdown(f'<div class="message-time">{timestamp}</div>', unsafe_allow_html=True)
    else:
        with col2:
            with st.container(border=False):
                st.markdown(f'<div class="chat-bubble chat-bubble-bot"><strong>Mental Health Guardian:</strong></div>', unsafe_allow_html=True)
                st.markdown(message)
                
                if confidence and matched_q:
                    st.markdown(f'<div class="confidence-score">Confidence: {confidence:.2%} | Matched: {matched_q}</div>', unsafe_allow_html=True)
                
                st.markdown(f'<div class="message-time">{timestamp}</div>', unsafe_allow_html=True)


# --- Main Application ---
def main():
    st.title("Mental Health Guardian")
    st.markdown("#### Your compassionate AI assistant for mental health questions")
    st.markdown("---")

    with st.sidebar:
        st.header("Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold:", min_value=0.0, max_value=1.0, value=0.55, step=0.05,
            help="Lower this value to get answers even if the model is less certain."
        )
        st.markdown("---")
        st.header("About")
        st.info("This chatbot is for informational purposes and is not a substitute for professional medical advice.")
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    df = load_data()
    model, question_embeddings = load_models()

    if df is None or model is None:
        st.warning("Chatbot is not operational due to missing files.")
        st.stop()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "bot",
                "message": "Hello! I'm your Mental Health Guardian. How can I support you today? Feel free to ask me anything about mental wellness.",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
        ]

    for chat in st.session_state.chat_history:
        display_chat_message(
            chat['role'], chat['message'], chat['timestamp'],
            chat.get('confidence'), chat.get('matched_question')
        )
    
    user_input = st.chat_input("Ask about stress, anxiety, therapy, etc.")
    
    if user_input:
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.chat_history.append({'role': 'user', 'message': user_input, 'timestamp': timestamp})
        
        with st.spinner("Finding the best information for you..."):
            response, confidence, matched_q = get_high_level_response(
                user_input, df, model, question_embeddings, threshold=confidence_threshold
            )
        
        st.session_state.chat_history.append({
            'role': 'bot', 'message': response, 'timestamp': timestamp,
            'confidence': confidence, 'matched_question': matched_q
        })
        st.rerun()

if __name__ == "__main__":
    main()