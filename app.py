import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional
from spellchecker import SpellChecker
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Mental Health Guardian",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Dark Theme & Custom CSS ---
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

# --- Model Loading with Caching ---
@st.cache_resource
def load_model():
    """Loads the sentence transformer model."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_data():
    """Loads the FAQ dataset."""
    try:
        return pd.read_csv('cleaned_mental_health_faq.csv')
    except Exception as e:
        st.warning(f"FAQ database not found.")
        return None

@st.cache_resource
def load_embeddings():
    """Loads pre-computed embeddings from pickle file."""
    try:
        with open('question_embeddings.npy', 'rb') as f:
            embeddings = np.load(f)
        return embeddings
    except Exception as e:
        try:
            with open('tfidf_vectorizer.pkl', 'rb') as f:
                embeddings = pickle.load(f)
            return embeddings
        except:
            st.warning(f"Embeddings file not found.")
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
        'motivation', 'confidence', 'relationships', 'loneliness', 'burnout', 'sadness',
        'fear', 'worry', 'self-harm', 'suicide', 'addiction', 'substance', 'abuse',
        'panic attack', 'insomnia', 'social anxiety', 'generalized anxiety'
    ]
    text_lower = text.lower()
    for term in mental_health_terms:
        if term in text_lower:
            keywords.append(term)
    return keywords

def get_supportive_response(keywords: list) -> str:
    """Provides a supportive response based on keywords."""
    topic = ', '.join(keywords) if keywords else 'mental health'
    responses = {
        'anxiety': "Anxiety is common and manageable. Deep breathing, mindfulness, exercise, and professional support can help. Consider reaching out to a mental health professional for personalized strategies.",
        'depression': "Depression is serious and treatable. Therapy, medication, lifestyle changes, and support from loved ones can help. Please reach out to a healthcare provider.",
        'stress': "Stress management involves exercise, meditation, time management, and talking to trusted people. Prioritize self-care and seek professional help if stress becomes overwhelming.",
        'sleep': "Good sleep is important for mental health. A consistent sleep schedule, relaxing bedtime routine, and comfortable environment help. Consult a healthcare professional if problems persist.",
        'trauma': "Trauma-focused therapy is very effective. Consider reaching out to a trauma specialist who can help you process and heal.",
        'panic': "Panic attacks are manageable with controlled breathing, grounding exercises, and professional therapy. Seek support from a mental health professional.",
        'meditation': "Meditation reduces anxiety and improves focus. Start with a few minutes daily and gradually increase your practice.",
        'relationships': "Healthy relationships need communication, boundaries, and respect. A counselor can help with relationship challenges.",
        'loneliness': "Connection through activities, communities, or professional support helps. Reaching out is a sign of strength.",
        'burnout': "Burnout recovery involves boundaries, breaks, hobbies, and support. Talk to a professional about recovery strategies.",
        'fear': "Fear is normal. Gradual exposure, cognitive behavioral therapy, and relaxation methods help. Professional support is available.",
        'ocd': "Obsessive-Compulsive Disorder responds well to Cognitive Behavioral Therapy and medication. Consult a mental health professional.",
        'adhd': "ADHD can be managed through therapy, medication, and lifestyle modifications. A healthcare provider can help determine the best approach.",
    }
    
    for key, response in responses.items():
        if key in topic.lower():
            return response
    
    return f"Your mental health matters. Professional support tailored to your situation can help. Consider connecting with a mental health professional."

# --- Core Chatbot Logic ---
def clean_text(text: str) -> str:
    """Cleans formatting issues from text."""
    if not text:
        return text
    text = text.replace('/p>', '')
    text = text.replace('<p>', '')
    text = text.replace('<p', '')
    text = text.replace('</p>', '')
    text = text.replace('&nbsp;', ' ')
    text = text.strip()
    return text

def get_response(user_query: str, df: Optional[pd.DataFrame], model: SentenceTransformer, 
                 embeddings: Optional[np.ndarray], threshold: float = 0.40) -> Tuple[str, float, Optional[str]]:
    """Generates a response with intelligent matching."""
    if not user_query or user_query.isspace():
        return "Please share what's on your mind. I'm here to help.", 0.0, None

    normalized_query = user_query.lower().strip()
    greetings = ['hi', 'hello', 'hey', 'greetings', 'howdy', 'hiya', 'hi there']
    farewells = ['bye', 'goodbye', 'see you', 'take care', 'see ya', 'farewell', 'bye bye']

    if normalized_query in greetings:
        return "Hello! I'm your Mental Health Guardian. What would you like to know about today?", 1.0, "Greeting"
    if normalized_query in farewells:
        return "Take care of yourself. Seeking help is a sign of strength. Feel free to reach out anytime.", 1.0, "Farewell"

    spell_checker = get_spell_checker()
    corrected_query = correct_spelling(normalized_query, spell_checker)
    keywords = extract_keywords(corrected_query)

    query_embedding = model.encode([corrected_query])
    
    # Try FAQ matching if we have data and embeddings
    if df is not None and embeddings is not None and len(embeddings) > 0 and embeddings.shape[0] > 0:
        try:
            similarities = cosine_similarity(query_embedding, embeddings).flatten()
            best_match_idx = similarities.argmax()
            best_similarity = float(similarities[best_match_idx])

            if best_similarity >= threshold:
                matched_q = df['Questions'].iloc[best_match_idx]
                answer = df['Answers'].iloc[best_match_idx]
                answer = clean_text(answer)
                intro = f"Based on your question about '{matched_q.strip()}', here's helpful information:\n\n"
                outro = "\n\nDisclaimer: This is for informational purposes only. Please consult a professional for personalized advice."
                return intro + answer + outro, best_similarity, "FAQ Database"
        except Exception as e:
            pass
    
    # Fallback: Supportive Response
    response = get_supportive_response(keywords)
    return response, 0.65, "Knowledge Base"

def stream_response(text: str):
    """Yields words to simulate a typing effect."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)

# --- Sidebar ---
def render_sidebar():
    """Renders the sidebar with app information and resources."""
    with st.sidebar:
        st.markdown("## Mental Health Guardian")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This AI chatbot provides:
        - Information on mental health topics
        - Support and coping strategies
        - Resources and guidance
        
        **Important:** This is NOT a replacement for professional mental health care.
        """)
        
        st.markdown("---")
        st.markdown("### Crisis Resources")
        st.markdown("""
        If you're in crisis:
        
        **National Suicide Prevention Lifeline**
        - Phone: 1-800-273-8255
        - Text: HOME to 741741
        
        **International**
        - Visit: findahelpline.com
        """)
        
        st.markdown("---")
        st.markdown("### How to Use")
        st.markdown("""
        1. Ask freely about your concerns
        2. Get helpful information
        3. Use resources provided
        4. Seek professional help when needed
        
        Tips:
        - Be specific about your concerns
        - More detail helps better responses
        - Ask follow-up questions anytime
        """)
        
        st.markdown("---")
        st.markdown("### Topics")
        topics = ["Anxiety", "Depression", "Stress", "Sleep", "Trauma", "Relationships", 
                 "Burnout", "Self-Care", "Meditation", "Coping", "Work-Life Balance", "Grief"]
        
        col1, col2 = st.columns(2)
        for i, topic in enumerate(topics):
            if i % 2 == 0:
                col1.markdown(f"- {topic}")
            else:
                col2.markdown(f"- {topic}")
        
        st.markdown("---")
        st.markdown("""
        Made with care for your mental health
        
        Remember: Taking care of yourself is not selfish. You deserve support.
        """)

# --- Main Application ---
def main():
    render_sidebar()
    
    st.title("Mental Health Chatbot")
    st.markdown("Your AI companion for mental health support and information")

    model = load_model()
    if model is None:
        st.error("Models failed to load. Please check dependencies.")
        st.stop()

    df = load_data()
    embeddings = load_embeddings()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": "Welcome! I'm your Mental Health Guardian. I'm here to provide information, support, and guidance on mental health topics. How can I help you today?",
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
                    Copy Response
                </button>
                """, unsafe_allow_html=True)

    # Suggested Questions
    if len(st.session_state.chat_history) == 1:
        st.markdown("---")
        st.subheader("Try asking about:")
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
            with st.spinner("Thinking..."):
                response, confidence, source = get_response(
                    prompt, df, model, embeddings, threshold=0.40
                )
            
            streamed_message = st.write_stream(stream_response(response))
            
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
