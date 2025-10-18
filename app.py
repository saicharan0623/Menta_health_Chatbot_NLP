import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import textwrap
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional
from spellchecker import SpellChecker # New import for spell correction

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
    /* Style for chat messages created by st.chat_message */
    [data-testid="stChatMessage"] {
        background-color: #374151;
        border-radius: 0.75rem;
        padding: 1rem;
    }
    /* Style for user messages */
    [data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"] p:contains("You:")) {
        background: linear-gradient(to right, #3b82f6, #60a5fa);
        color: #ffffff;
    }
    /* General text and headers */
    h1, h2, h3 { color: #60a5fa; }
    /* Copy button style */
    .copy-button {
        background-color: #4b5563;
        color: #e5e7eb;
        border: none;
        padding: 5px 10px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 0.8rem;
        margin-top: 10px;
    }
    .copy-button:hover { background-color: #6b7280; }
</style>
""", unsafe_allow_html=True)

# --- JavaScript for Copy-to-Clipboard ---
st.html("""
<script>
function copyToClipboard(elementId) {
    var copyText = document.getElementById(elementId).innerText;
    navigator.clipboard.writeText(copyText);
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
        st.error(f"Error loading models: {e}. Ensure 'question_embeddings.npy' is present.")
        return None, None

@st.cache_data
def load_data() -> Optional[pd.DataFrame]:
    """Loads the FAQ dataset."""
    try:
        return pd.read_csv('cleaned_mental_health_faq.csv')
    except Exception as e:
        st.error(f"Error loading data: {e}. Ensure 'cleaned_mental_health_faq.csv' is present.")
        return None

# --- NEW: Spell Correction Function ---
@st.cache_resource
def get_spell_checker():
    """Initializes and returns a SpellChecker object."""
    return SpellChecker()

def correct_spelling(text: str, checker: SpellChecker) -> str:
    """Corrects spelling mistakes in the input text."""
    words = text.split()
    corrected_words = [checker.correction(word) for word in words]
    # checker.correction can return None, so we handle that case
    return " ".join(c for c in corrected_words if c is not None)


# --- Core Chatbot Logic ---
def get_response(user_query: str, df: pd.DataFrame, model: SentenceTransformer, embeddings: np.ndarray, threshold: float) -> Tuple[str, float, Optional[str]]:
    """Generates a response, now with spell correction."""
    if not user_query or user_query.isspace():
        return "Please ask a question.", 0.0, None

    # --- Step 1: Handle Greetings and Farewells (as before) ---
    normalized_query = user_query.lower().strip()
    greetings = ['hi', 'hello', 'hey']
    farewells = ['bye', 'goodbye', 'see you']

    if normalized_query in greetings:
        return "Hello there! I'm here to help. What's on your mind?", 1.0, "Greeting"
    if normalized_query in farewells:
        return "Goodbye! Remember to take care of yourself.", 1.0, "Farewell"

    # --- Step 2: Correct Spelling in the User's Query ---
    spell_checker = get_spell_checker()
    corrected_query = correct_spelling(normalized_query, spell_checker)

    # --- Step 3: Find the Best Match using the Corrected Query ---
    query_embedding = model.encode([corrected_query])
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    best_match_idx = similarities.argmax()
    best_similarity = similarities[best_match_idx]

    if best_similarity >= threshold:
        matched_q = df['Questions'].iloc[best_match_idx]
        answer = df['Answers'].iloc[best_match_idx]
        intro = f"Based on your question, which I understand as being about **'{matched_q.strip()}'**, here's some information:\n\n"
        outro = "\n\n*Disclaimer: This is for informational purposes only. Please consult a professional for personal advice.*"
        return intro + answer + outro, best_similarity, matched_q
    else:
        return textwrap.dedent("""
            I'm sorry, I couldn't find a direct answer. Could you try rephrasing your question?
            If you need immediate support, please contact a healthcare professional.
        """), 0.0, None

def stream_response(text: str):
    """Yields words to simulate a typing effect."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.04)

# --- Main Application ---
def main():
    st.title("Mental Health Guardian")

    df = load_data()
    model, question_embeddings = load_models()

    if df is None or model is None:
        st.warning("Chatbot is not fully operational due to missing files.")
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": "Hello! I'm your AI Guardian for mental health questions. How can I support you today?",
            "timestamp": datetime.now()
        }]

    # --- Display Chat History using st.chat_message ---
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            # For the user, we add a simple prefix.
            if chat["role"] == "user":
                st.markdown(f"**You:**\n{chat['content']}")
            else:
                # For the bot, we display the message and a copy button
                message_id = f"msg_{int(chat['timestamp'].timestamp())}"
                st.markdown(f"<div id='{message_id}'>{chat['content']}</div>", unsafe_allow_html=True)

                st.markdown(f"""
                <button class="copy-button" onclick="copyToClipboard('{message_id}')">
                    Copy Text
                </button>
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
    prompt = st.chat_input("Ask about stress, anxiety, etc.")
    
    if "suggestion_clicked" in st.session_state and st.session_state.suggestion_clicked:
        prompt = st.session_state.suggestion_clicked
        st.session_state.suggestion_clicked = None

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt, "timestamp": datetime.now()})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(f"**You:**\n{prompt}")

        # Get and display bot response with streaming
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, _, _ = get_response(prompt, df, model, question_embeddings, 0.55)
            
            streamed_message = st.write_stream(stream_response(response))
            
        st.session_state.chat_history.append({"role": "assistant", "content": streamed_message, "timestamp": datetime.now()})
        st.rerun()

if __name__ == "__main__":
    main()
