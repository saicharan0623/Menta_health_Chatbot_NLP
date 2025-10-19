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
import warnings
warnings.filterwarnings('ignore')

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
    .sidebar-card {
        background-color: #1e293b;
        border-left: 4px solid #3b82f6;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
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
def load_advanced_models() -> Tuple[Optional[SentenceTransformer], Optional[pipeline]]:
    """Loads advanced sentence transformer model and text generation pipeline."""
    try:
        # Using a more powerful model for better semantic understanding
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Text generation pipeline for fallback responses
        generator = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)
        
        return embedding_model, generator
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

@st.cache_data
def load_data() -> Optional[pd.DataFrame]:
    """Loads the FAQ dataset."""
    try:
        return pd.read_csv('cleaned_mental_health_faq.csv')
    except Exception as e:
        st.warning(f"FAQ database not found: {e}. AI will generate responses.")
        return None

@st.cache_data
def compute_embeddings(df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    """Computes embeddings for all questions in the dataset."""
    if df is None or len(df) == 0:
        return np.array([])
    
    try:
        questions = df['Questions'].tolist() if 'Questions' in df.columns else []
        if len(questions) == 0:
            return np.array([])
        
        embeddings = model.encode(questions, show_progress_bar=False)
        return embeddings
    except Exception as e:
        st.warning(f"Could not compute embeddings: {e}")
        return np.array([])

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
        'motivation', 'confidence', 'relationships', 'loneliness', 'burnout', 'sadness',
        'fear', 'worry', 'self-harm', 'suicide', 'addiction', 'substance', 'abuse'
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
    
    prompt = f"Provide helpful, empathetic mental health advice about: {', '.join(keywords[:3])}. Keep it concise, supportive and practical."
    
    try:
        response = generator(prompt, max_length=250, num_beams=2)
        generated_text = response[0]['generated_text']
        return generated_text + "\n\n*This response was AI-generated based on your question. Please consult a professional for personalized advice.*"
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
        'burnout': "Burnout is a state of physical and emotional exhaustion. Recovery involves setting boundaries, taking breaks, pursuing hobbies, and seeking support. Don't hesitate to talk to a professional about effective recovery strategies.",
        'fear': "Fear is a normal emotion, but when it becomes overwhelming, it's important to address it. Techniques like gradual exposure, cognitive behavioral therapy, and relaxation methods can help. Professional support is available if fear is limiting your life.",
    }
    
    for key, response in responses.items():
        if key in topic.lower():
            return response
    
    return f"Thank you for reaching out about {topic}. Your mental health and well-being are important. Professional support tailored to your specific situation can be incredibly helpful. Consider connecting with a mental health professional who can provide personalized guidance."

# --- Core Chatbot Logic with Advanced Matching ---
def get_response(user_query: str, df: Optional[pd.DataFrame], model: SentenceTransformer, 
                 generator: pipeline, embeddings: np.ndarray, 
                 threshold: float = 0.40) -> Tuple[str, float, Optional[str]]:
    """Generates a response with advanced matching and fallback generation."""
    if not user_query or user_query.isspace():
        return "Please share what's on your mind. I'm here to help.", 0.0, None

    # Handle Greetings and Farewells
    normalized_query = user_query.lower().strip()
    greetings = ['hi', 'hello', 'hey', 'greetings', 'howdy', 'hiya']
    farewells = ['bye', 'goodbye', 'see you', 'take care', 'see ya', 'farewell']

    if normalized_query in greetings:
        return "Hello! 👋 I'm your Mental Health Guardian. What would you like to know about today?", 1.0, "Greeting"
    if normalized_query in farewells:
        return "Take care of yourself! 💚 Remember, seeking help is a sign of strength. Feel free to reach out anytime.", 1.0, "Farewell"

    # Spell Correction
    spell_checker = get_spell_checker()
    corrected_query = correct_spelling(normalized_query, spell_checker)
    
    # Extract keywords
    keywords = extract_keywords(corrected_query)

    # Find Best Match using Advanced Embeddings
    query_embedding = model.encode([corrected_query])
    
    # Try FAQ matching if we have data and embeddings
    if df is not None and len(embeddings) > 0 and embeddings.shape[0] > 0:
        try:
            similarities = cosine_similarity(query_embedding, embeddings).flatten()
            best_match_idx = similarities.argmax()
            best_similarity = float(similarities[best_match_idx])

            if best_similarity >= threshold:
                matched_q = df['Questions'].iloc[best_match_idx]
                answer = df['Answers'].iloc[best_match_idx]
                intro = f"Based on your question about **'{matched_q.strip()}'**, here's helpful information:\n\n"
                outro = "\n\n*Disclaimer: This is for informational purposes only. Please consult a professional for personalized advice.*"
                return intro + answer + outro, best_similarity, "FAQ Database"
        except Exception as e:
            st.debug(f"Error in FAQ matching: {e}")
    
    # Fallback: Generate Response using AI
    if generator is not None:
        try:
            generated_response = generate_fallback_response(corrected_query, generator, keywords)
            return generated_response, 0.75, "AI Generated"
        except:
            pass
    
    # Ultimate Fallback: Supportive Default
    default_response = get_supportive_default_response(keywords)
    return default_response, 0.65, "Supportive Response"

def stream_response(text: str):
    """Yields words to simulate a typing effect."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)

# --- Sidebar with Details ---
def render_sidebar():
    """Renders the sidebar with app information and resources."""
    with st.sidebar:
        st.markdown("## 🛡️ Mental Health Guardian Pro")
        
        st.markdown("---")
        st.markdown("### 📊 About This App")
        st.markdown("""
        This AI-powered chatbot provides:
        - **Information** on mental health topics
        - **Support** and coping strategies
        - **Resources** and guidance
        
        **⚠️ Important:** This is NOT a replacement for professional mental health care.
        """)
        
        st.markdown("---")
        st.markdown("### 🆘 Crisis Resources")
        st.markdown("""
        If you're in crisis, please reach out:
        
        **National Suicide Prevention Lifeline**
        - 📞 1-800-273-8255
        - 💬 Text HOME to 741741
        
        **Crisis Text Line**
        - Text: HOME to 741741
        
        **International**
        - 🌍 findahelpline.com
        """)
        
        st.markdown("---")
        st.markdown("### 💡 How to Use")
        st.markdown("""
        1. **Ask freely** - Share your concerns
        2. **Get info** - Receive helpful information
        3. **Take action** - Use resources provided
        4. **Seek help** - Contact professionals when needed
        
        ✅ **Tips:**
        - Be specific about your concerns
        - The more detail, the better the response
        - Ask follow-up questions anytime
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Topics I Can Help With")
        topics = ["Anxiety", "Depression", "Stress", "Sleep Issues", "Trauma", "Relationships", 
                 "Burnout", "Self-Care", "Meditation", "Coping Skills", "Work-Life Balance", "Grief"]
        
        col1, col2 = st.columns(2)
        for i, topic in enumerate(topics):
            if i % 2 == 0:
                col1.markdown(f"• {topic}")
            else:
                col2.markdown(f"• {topic}")
        
        st.markdown("---")
        st.markdown("### ℹ️ App Details")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Type", "Advanced AI")
            st.metric("Response Mode", "Dynamic")
        with col2:
            st.metric("Language", "English")
            st.metric("Version", "2.0 Pro")
        
        st.markdown("---")
        st.markdown("""
        **Made with ❤️ for your mental health**
        
        *Remember: Taking care of yourself is not selfish. You deserve support.*
        """)

# --- Main Application ---
def main():
    render_sidebar()
    
    st.title("🛡️ Mental Health Guardian Pro")
    st.markdown("*Your AI companion for mental health support and information*")

    model, generator = load_advanced_models()

    if model is None:
        st.error("⚠️ Advanced models failed to load. Please check dependencies.")
        st.stop()

    df = load_data()
    embeddings = compute_embeddings(df, model) if df is not None else np.array([])

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
            "How to improve sleep?",
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
                    prompt, df, model, generator, embeddings, threshold=0.40
                )
            
            streamed_message = st.write_stream(stream_response(response))
            
            # Show confidence badge
            if confidence > 0:
                st.markdown(f"""
                <div class="confidence-badge">
                    Confidence: {confidence:.1%} | Source: {source or 'Generated'}
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
