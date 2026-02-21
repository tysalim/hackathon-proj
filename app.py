import streamlit as st
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
from model_utilities import load_model, predict_grade_level
import torch
import re

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Readability Toolkit", page_icon="📘", layout="wide")

# -----------------------------
# Load Models (Cached)
# -----------------------------
@st.cache_resource(show_spinner=True)
def get_grade_model():
    return load_model()

@st.cache_resource(show_spinner=True)
def get_simplifier():
    model_name = "eilamc14/bart-base-text-simplification"
    try:
        simplifier = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        st.session_state["pipeline_type"] = "text2text"
        return simplifier
    except KeyError:
        # Fallback for older transformers
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)

        def simplifier(text, max_length=512):
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=max_length)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.session_state["pipeline_type"] = "fallback"
        return simplifier

# Load cached models
model, tfidf = get_grade_model()
simplifier = get_simplifier()

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a tool:", ["Grade Level Classifier", "Text Simplifier"])

# -----------------------------
# Utility Functions
# -----------------------------
def clean_output(text):
    """Clean and format simplified text"""
    # Remove repeated words
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    # Replace multiple punctuation
    text = re.sub(r'([.!?])\1+', r'\1', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Capitalize sentences
    sentences = re.split(r'([.!?])', text)
    cleaned = ""
    for i in range(0, len(sentences)-1, 2):
        s = sentences[i].strip()
        p = sentences[i+1]
        if s:
            s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
            cleaned += s + p + " "
    return cleaned.strip()

def build_prompt(text, grade):
    """Build strict, dynamic prompt based on target reading grade"""
    prompt = f"Simplify the following text for a student at grade {grade} reading level.\n"
    prompt += "- Rewrite entirely using your own words. Do NOT copy the original text.\n"

    if grade <= 4:
        prompt += (
            "- Use very short sentences (max 5 words each)\n"
            "- Use only the 1000 most common English words\n"
            "- Avoid any complex or abstract words\n"
            "- Split long sentences into multiple short sentences\n"
            "- Keep meaning accurate but extremely simple\n"
        )
    elif grade <= 8:
        prompt += (
            "- Use short sentences (max 8 words)\n"
            "- Use mostly common words\n"
            "- Avoid advanced vocabulary\n"
            "- Keep ideas clear and simple\n"
        )
    else:
        prompt += (
            "- Use clear sentences (max 12 words)\n"
            "- Mostly common words, some advanced words allowed\n"
            "- Keep meaning accurate and easy to understand\n"
        )
    prompt += f"Text: {text}"
    return prompt


# PAGE 1 — Grade Level Classifier

if page == "Grade Level Classifier":
    st.title("Reading Grade Level Classifier")
    text_input = st.text_area("Enter text to classify:", height=200)

    if st.button("Predict Grade Level"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            grade = predict_grade_level(text_input, model, tfidf)
            st.success(f"Predicted Grade Level: **Grade {grade}**")

            if grade <= 4:
                st.info("This text is suitable for elementary school level.")
            elif grade <= 8:
                st.info("This text is suitable for middle school level.")
            else:
                st.info("This text is suitable for high school level or above.")

# -----------------------------
# PAGE 2 — Text Simplifier
# -----------------------------
elif page == "Text Simplifier":
    st.title("Text Simplifier")
    text_input = st.text_area("Enter text to simplify:", height=200)
    target_grade = st.slider("Select target reading grade level:", 1, 12, 6)

    if st.button("Simplify Text"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Simplifying text..."):
                prompt = build_prompt(text_input, target_grade)

                # Run through pipeline
                if st.session_state.get("pipeline_type") == "text2text":
                    raw_output = simplifier(prompt, max_length=256, truncation=True)[0]["generated_text"]
                else:
                    raw_output = simplifier(prompt, max_length=256)

                # Clean and format
                result = clean_output(raw_output)

            st.subheader("Simplified Text")
            st.markdown(f"{result}")