import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import textwrap
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional
from spellchecker import SpellChecker
import re
from transformers import pipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="Mental Health Guardian Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Advanced Dark Theme & Custom CSS ---
st.markdown("""
<style>
    .stApp {
        background-image: linear-gradient(180deg, #0f172a, #1e293b);
        color: #e2e8f0;
    }
    [data-testid="stChatMessage"] {
        background-color: #1e293b;
        border-radius: 0.75rem;
        padding: 1rem;
        border-left: 4px solid #3b82f6;
    }
    h1, h2, h3 { color: #60a5fa; }
    .copy-button {
        background-color: #334155;
        color: #e2e8f0;
        border: none;
        padding: 8px 12px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 0.8rem;
        margin-top: 10px;
        transition: 0.3s;
    }
    .copy-button:hover { 
        background-color: #475569;
        box-shadow: 0 0 8px rgba(59, 130, 246, 0.5);
    }
    .confidence-badge {
        display: inline-block;
        background-color: #10b981;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        margin-top: 8px;
    }
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

# --- Advanced Model Loading with Caching ---
@st.cache_resource
def load_advanced_models() -> Tuple[Optional[SentenceTransformer], Optional[pipeline], Optional[np.ndarray]]:
    """Loads advanced sentence transformer model and text generation pipeline."""
    try:
        # Using a more powerful model for better semantic understanding
        embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Load pre-computed embeddings if available
        try:
            question_embeddings = np.load('question_embeddings.npy')
        except:
            question_embeddings = None
        
        # Text generation pipeline for fallback responses
        generator = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)
        
        return embedding_model, generator, question_embeddings
    except Exception as e:
        st.error(f"Error loading models: {e}. Ensure required models are available.")
        return None, None, None

@st.cache_data
def load_data() -> Optional[pd.DataFrame]:
    """Loads the FAQ dataset."""
    try:
        return pd.read_csv('cleaned_mental_health_faq.csv')
    except Exception as e:
        st.error(f"Error loading data: {e}. Ensure 'cleaned_mental_health_faq.csv' is present.")
        return None

@st.cache_resource
def get_spell_checker():
    """Initializes and returns a SpellChecker object."""
    return SpellChecker()

def correct_spelling(text: str, checker: SpellChecker) -> str:
    """Corrects spelling mistakes in the input text."""
    words = text.split()
    corrected_words = [checker.correction(word) if checker.correction(word) else word for word in words]
    return " ".join(corrected_words)

def extract_keywords(text: str) -> list:
    """Extracts key mental health terms from user query."""
    keywords = []
    mental_health_terms = [
        'anxiety', 'depression', 'stress', 'sleep', 'trauma', 'panic', 'phobia',
        'ocd', 'bipolar', 'schizophrenia', 'ptsd', 'adhd', 'autism', 'eating',
        'anger', 'grief', 'mindfulness', 'meditation', 'therapy', 'coping',
        'motivation', 'confidence', 'relationships', 'loneliness', 'burnout'
    ]
    text_lower = text.lower()
    for term in mental_health_terms:
        if term in text_lower:
            keywords.append(term)
    return keywords

def generate_fallback_response(user_query: str, generator: pipeline, keywords: list) -> str:
    """Generates a thoughtful response using AI when no direct match is found."""
    if not keywords:
        keywords = ["mental health and wellbeing"]
    
    prompt = f"Provide helpful, empathetic mental health advice about: {', '.join(keywords)}. Keep it concise and supportive."
    
    try:
        response = generator(prompt, max_length=200, num_beams=4)
        generated_text = response[0]['generated_text']
        return generated_text + "\n\n*This response was generated based on your question. Please consult a professional for personalized advice.*"
    except:
        return get_supportive_default_response(keywords)

def get_supportive_default_response(keywords: list) -> str:
    """Provides a supportive default response when generation fails."""
    topic = ', '.join(keywords) if keywords else 'mental health'
    responses = {
        'anxiety': "Anxiety is a common experience that many people face. It's important to remember that anxiety can be managed through various techniques such as deep breathing, mindfulness, exercise, and professional support. If you're struggling, reaching out to a mental health professional can provide personalized strategies.",
        'depression': "Depression is a serious mental health condition that deserves proper attention. Many people find relief through therapy, medication, lifestyle changes, and support from loved ones. If you're experiencing depression, please reach out to a healthcare provider or mental health professional.",
        'stress': "Stress is a natural part of life, but managing it effectively is important. Strategies like exercise, meditation, time management, and talking to someone you trust can help. Remember to prioritize self-care and reach out for professional help if stress becomes overwhelming.",
        'sleep': "Good sleep is crucial for mental and physical health. Establishing a consistent sleep schedule, creating a relaxing bedtime routine, and maintaining a comfortable sleep environment can improve sleep quality. If sleep problems persist, consult a healthcare professional.",
        'trauma': "Trauma can have lasting effects on mental health. Professional support through trauma-focused therapy is often very effective. If you're dealing with trauma, consider reaching out to a trauma specialist or counselor who can help you process and heal.",
        'meditation': "Meditation is a powerful tool for mental wellness. Regular practice can reduce anxiety, improve focus, and enhance overall well-being. Start with just a few minutes daily and gradually increase your practice.",
        'relationships': "Healthy relationships are important for mental health. Communication, boundaries, and mutual respect are key. If you're facing relationship challenges, talking to a counselor or therapist can provide valuable insights and strategies.",
        'loneliness': "Loneliness can impact mental health, but there are ways to address it. Connecting with others through activities, communities, or professional support can help. Remember that reaching out is a sign of strength.",
    }
    
    for key, response in responses.items():
        if key in topic.lower():
            return response
    
    return f"Thank you for asking about {topic}. Your mental health and well-being are important. If you'd like specific guidance, please consider consulting with a mental health professional who can provide personalized support tailored to your needs."

# --- Core Chatbot Logic with Advanced Matching ---
def get_response(user_query: str, df: pd.DataFrame, model: SentenceTransformer, 
                 generator: pipeline, embeddings: Optional[np.ndarray], 
                 threshold: float = 0.45) -> Tuple[str, float, Optional[str]]:
    """Generates a response with advanced matching and fallback generation."""
    if not user_query or user_query.isspace():
        return "Please share what's on your mind. I'm here to help.", 0.0, None

    # Handle Greetings and Farewells
    normalized_query = user_query.lower().strip()
    greetings = ['hi', 'hello', 'hey', 'greetings', 'howdy']
    farewells = ['bye', 'goodbye', 'see you', 'take care', 'see ya']

    if normalized_query in greetings:
        return "Hello! I'm your Mental Health Guardian. What would you like to know about today?", 1.0, "Greeting"
    if normalized_query in farewells:
        return "Take care of yourself! Remember, seeking help is a sign of strength. Feel free to reach out anytime.", 1.0, "Farewell"

    # Spell Correction
    spell_checker = get_spell_checker()
    corrected_query = correct_spelling(normalized_query, spell_checker)
    
    # Extract keywords
    keywords = extract_keywords(corrected_query)

    # Find Best Match using Advanced Embeddings
    query_embedding = model.encode([corrected_query])
    
    if embeddings is not None and len(embeddings) > 0:
        similarities = cosine_similarity(query_embedding, embeddings).flatten()
        best_match_idx = similarities.argmax()
        best_similarity = similarities[best_match_idx]

        # Lower threshold for better matching
        if best_similarity >= threshold and df is not None:
            matched_q = df['Questions'].iloc[best_match_idx]
            answer = df['Answers'].iloc[best_match_idx]
            intro = f"Based on your question about **'{matched_q.strip()}'**, here's helpful information:\n\n"
            outro = "\n\n*Disclaimer: This is for informational purposes only. Please consult a professional for personalized advice.*"
            confidence = f"Match confidence: {best_similarity:.1%}"
            return intro + answer + outro, best_similarity, matched_q
    
    # Fallback: Generate Response using AI
    if generator is not None:
        generated_response = generate_fallback_response(corrected_query, generator, keywords)
        return generated_response, 0.6, "Generated Response"
    
    # Ultimate Fallback: Supportive Default
    default_response = get_supportive_default_response(keywords)
    return default_response, 0.5, "Supportive Response"

def stream_response(text: str):
    """Yields words to simulate a typing effect."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)

# --- Main Application ---
def main():
    st.title("🛡️ Mental Health Guardian Pro")
    st.markdown("*Your AI companion for mental health support and information*")

    model, generator, question_embeddings = load_advanced_models()
    df = load_data()

    if model is None:
        st.error("⚠️ Advanced models failed to load. Please check dependencies.")
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": "👋 Welcome! I'm your Mental Health Guardian Pro. I'm here to provide information, support, and guidance on mental health topics. How can I help you today?",
            "timestamp": datetime.now()
        }]

    # Display Chat History
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            if chat["role"] == "user":
                st.markdown(f"**You:**\n{chat['content']}")
            else:
                message_id = f"msg_{int(chat['timestamp'].timestamp())}"
                st.markdown(f"<div id='{message_id}'>{chat['content']}</div>", unsafe_allow_html=True)
                st.markdown(f"""
                <button class="copy-button" onclick="copyToClipboard('{message_id}')">
                    📋 Copy Response
                </button>
                """, unsafe_allow_html=True)

    # Suggested Questions
    if len(st.session_state.chat_history) == 1:
        st.markdown("---")
        st.subheader("💡 Try asking about:")
        cols = st.columns(3)
        suggestions = [
            "What is anxiety?",
            "How can I improve sleep?",
            "How to manage stress?"
        ]
        for i, suggestion in enumerate(suggestions):
            if cols[i].button(suggestion, use_container_width=True, key=f"btn_{i}"):
                st.session_state.suggestion_clicked = suggestion
                st.rerun()

    # User Input
    prompt = st.chat_input("Ask about mental health, stress, anxiety, or wellbeing...")
    
    if "suggestion_clicked" in st.session_state and st.session_state.suggestion_clicked:
        prompt = st.session_state.suggestion_clicked
        st.session_state.suggestion_clicked = None

    if prompt:
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now()
        })
        
        with st.chat_message("user"):
            st.markdown(f"**You:**\n{prompt}")

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response, confidence, source = get_response(
                    prompt, df, model, generator, question_embeddings, threshold=0.45
                )
            
            streamed_message = st.write_stream(stream_response(response))
            
            # Show confidence badge
            if confidence > 0:
                st.markdown(f"""
                <div class="confidence-badge">
                    Confidence: {confidence:.1%} | Source: {source or 'AI Generated'}
                </div>
                """, unsafe_allow_html=True)
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": streamed_message,
            "timestamp": datetime.now()
        })
        st.rerun()

if __name__ == "__main__":
    main()
