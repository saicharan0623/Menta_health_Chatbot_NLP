import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import re
import html
import textwrap
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional

# --- Page Configuration ---
st.set_page_config(
    page_title="Mental Health Guardian",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Advanced Dark Theme & Custom CSS ---
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
    h1, h2, h3 { color: #60a5fa; }

    /* --- Chat Bubble Styling --- */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .chat-message {
        display: flex;
        align-items: flex-start;
        margin-bottom: 1.5rem;
        animation: fadeIn 0.5s ease-out;
    }

    .chat-message.user { justify-content: flex-end; }
    .chat-message.bot { justify-content: flex-start; }

    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #4b5563;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 0.9rem;
        font-weight: bold;
        flex-shrink: 0;
    }

    .chat-message.user .avatar { margin-left: 1rem; background-color: #3b82f6; }
    .chat-message.bot .avatar { margin-right: 1rem; }

    .message-content {
        padding: 0.8rem 1.2rem;
        border-radius: 0.75rem;
        max-width: 70%;
        position: relative;
    }

    .chat-message.user .message-content {
        background: linear-gradient(to right, #3b82f6, #60a5fa);
        color: #ffffff;
        border-bottom-right-radius: 0;
    }

    .chat-message.bot .message-content {
        background: #374151;
        color: #e5e7eb;
        border-bottom-left-radius: 0;
    }

    .message-meta {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-top: 0.5rem;
    }

    .chat-message.user .message-meta { text-align: right; }
    .chat-message.bot .message-meta { text-align: left; }

    .confidence-score {
        font-size: 0.8rem; color: #d1d5db; font-style: italic;
        margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid #4b5563;
    }

    .copy-button {
        background: none; border: none; color: #9ca3af;
        cursor: pointer; font-size: 0.8rem;
        position: absolute; top: 5px; right: 8px;
        opacity: 0.5; transition: opacity 0.3s;
    }
    .message-content:hover .copy-button { opacity: 1; }
</style>
""", unsafe_allow_html=True)

# --- JavaScript for Copy-to-Clipboard ---
st.html("""
<script>
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        // action on success
    }, function(err) {
        console.error('Could not copy text: ', err);
    });
}
</script>
""")


# --- Caching and Model Loading ---
@st.cache_resource
def load_models() -> Tuple[Optional[SentenceTransformer], Optional[np.ndarray]]:
    """Loads the sentence transformer model and pre-computed embeddings."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        question_embeddings = np.load('question_embeddings.npy')
        return model, question_embeddings
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure 'question_embeddings.npy' is present.")
        return None, None

@st.cache_data
def load_data() -> Optional[pd.DataFrame]:
    """Loads the FAQ dataset."""
    try:
        return pd.read_csv('cleaned_mental_health_faq.csv')
    except Exception as e:
        st.error(f"Error loading data: {e}. Please ensure 'cleaned_mental_health_faq.csv' is present.")
        return None

# --- Core Chatbot Logic ---
def get_response(user_query: str, df: pd.DataFrame, model: SentenceTransformer, embeddings: np.ndarray, threshold: float) -> Tuple[str, float, Optional[str]]:
    """Generates a response based on user query."""
    if not user_query or user_query.isspace():
        return "Please feel free to ask me anything about mental health.", 0.0, None

    normalized_query = user_query.lower().strip()
    greetings = ['hi', 'hello', 'hey', 'yo', 'greetings']
    farewells = ['bye', 'goodbye', 'see you', 'farewell', 'take care']

    if normalized_query in greetings:
        return "Hello there! I'm here to help. What's on your mind?", 1.0, "Greeting"
    if normalized_query in farewells:
        return "Goodbye! Remember to take care of yourself.", 1.0, "Farewell"

    query_embedding = model.encode([user_query])
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    best_match_idx = similarities.argmax()
    best_similarity = similarities[best_match_idx]

    if best_similarity >= threshold:
        matched_q = df['Questions'].iloc[best_match_idx]
        answer = df['Answers'].iloc[best_match_idx]
        intro = f"That's a great question about **'{matched_q.strip()}'**. Here's some information:\n\n"
        outro = "\n\n*Disclaimer: I am an AI assistant. Please consult with a healthcare professional for personal advice.*"
        return intro + answer + outro, best_similarity, matched_q
    else:
        response = textwrap.dedent("""
            I'm sorry, I couldn't find a direct answer to your question in my knowledge base.

            Could you try rephrasing it? Sometimes, a different wording can help me understand better.

            If you need immediate support, please reach out to a mental health professional or a crisis hotline. Your well-being is the top priority.
        """)
        return response, 0.0, None

def stream_response(text: str):
    """Yields words from a text string with a delay to simulate typing."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.04)

# --- Main Application ---
def main():
    st.title("Mental Health Guardian")

    df = load_data()
    model, question_embeddings = load_models()

    if df is None or model is None:
        st.warning("Chatbot is not fully operational. Please check error messages above.")
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{
            "role": "bot",
            "message": "Hello! I'm your AI Guardian for mental health questions. How can I support you today?",
            "timestamp": datetime.now(),
        }]

    # --- Chat History Display ---
    for chat in st.session_state.chat_history:
        role = chat["role"]
        message = chat["message"]
        timestamp = chat["timestamp"].strftime("%H:%M")

        avatar_text = "You" if role == "user" else "Bot"
        message_alignment = "user" if role == "user" else "bot"
        
        message_id = f"msg_{int(chat['timestamp'].timestamp())}"

        copy_button_html = f"""
            <button class="copy-button" onclick="copyToClipboard(document.getElementById('{message_id}').innerText)">
                Copy
            </button>
        """ if role == 'bot' else ''

        st.markdown(f"""
        <div class="chat-message {message_alignment}">
            {'<div class="message-content">' if role == 'user' else ''}
            <div class="avatar">{avatar_text}</div>
            <div class="message-content">
                {copy_button_html}
                <div id="{message_id}">{message}</div>
                <div class="message-meta">{timestamp}</div>
            </div>
            {'</div>' if role == 'user' else ''}
        </div>
        """, unsafe_allow_html=True)
    
    # --- Suggested Questions ---
    if len(st.session_state.chat_history) == 1:
        st.markdown("---")
        st.subheader("Or try one of these common questions:")
        cols = st.columns(3)
        suggestions = ["What is anxiety?", "How can I improve my sleep?", "What are signs of depression?"]
        for i, suggestion in enumerate(suggestions):
            if cols[i].button(suggestion, use_container_width=True):
                st.session_state.suggestion_clicked = suggestion
                st.rerun()

    # --- User Input ---
    prompt = st.chat_input("Ask about stress, anxiety, therapy, etc.")
    
    if "suggestion_clicked" in st.session_state and st.session_state.suggestion_clicked:
        prompt = st.session_state.suggestion_clicked
        st.session_state.suggestion_clicked = None

    if prompt:
        st.session_state.chat_history.append({"role": "user", "message": prompt, "timestamp": datetime.now()})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("bot", avatar="🤖"):
            with st.spinner("Bot is thinking..."):
                time.sleep(1)
                response, confidence, matched_q = get_response(prompt, df, model, question_embeddings, 0.55)
            
            streamed_message = st.write_stream(stream_response(response))
            
        st.session_state.chat_history.append({"role": "bot", "message": streamed_message, "timestamp": datetime.now()})
        st.rerun()

if __name__ == "__main__":
    main()
