# Mental Health Guardian

A compassionate AI assistant for mental health questions powered by Streamlit and transformer models.

## Features
- Semantic search using sentence transformers
- Empathetic responses to mental health questions
- Confidence scoring for answer accuracy
- Dark theme UI

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/mental-health-guardian.git
cd mental-health-guardian
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run main.py
```

## Required Files
- `cleaned_mental_health_faq.csv` - FAQ dataset
- `question_embeddings.npy` - Pre-computed embeddings

## Usage
Run the application and ask mental health-related questions. The chatbot will provide informative responses based on the FAQ database.

## Disclaimer
This chatbot is for informational purposes only and is not a substitute for professional medical advice.
```

**4. .gitignore** - Ignore unnecessary files:
```
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env
.DS_Store
*.csv
*.npy
.streamlit/
