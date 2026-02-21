import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import re

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Readability Toolkit", page_icon="📘", layout="wide")

# -----------------------------
# Cached Models
# -----------------------------


@st.cache_resource(show_spinner=True)
def get_simplifier():
    model_name = "google/flan-t5-small"

    # Explicitly load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Create the pipeline manually
    simplifier_pipeline = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    return simplifier_pipeline

# Load models
model, tfidf = get_grade_model()
simplifier = get_simplifier()

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a tool:", ["Grade Level Classifier", "Text Simplifier"])

# -----------------------------
# PAGE 1 — Grade Level Classifier
# -----------------------------
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

    def clean_output(text):
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

    if st.button("Simplify Text"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Simplifying text..."):
                # Dynamic strictness based on grade
                if target_grade <= 4:
                    strictness = (
                        "- Use extremely simple words only\n"
                        "- Short sentences ≤6 words\n"
                        "- No complex punctuation\n"
                        "- Rephrase ideas, do not copy text\n"
                        "- Remove any extra clauses"
                    )
                elif target_grade <= 8:
                    strictness = (
                        "- Use simple and common words\n"
                        "- Sentences ≤10 words\n"
                        "- Keep ideas clear and rephrased\n"
                        "- Avoid copying original sentence structure"
                    )
                else:
                    strictness = (
                        "- Use clear and concise wording\n"
                        "- Moderate sentence length ≤15 words\n"
                        "- Rephrase complex ideas\n"
                        "- Avoid verbatim copying"
                    )

                # Construct FLAN-T5 instruction prompt
                prompt = (
                    f"Simplify the following text for a grade {target_grade} student.\n"
                    f"{strictness}\n\n"
                    f"Text: {text_input}"
                )

                # Generate simplified text
                raw_output = simplifier(prompt, max_length=512)[0]["generated_text"]
                result = clean_output(raw_output)

            # Display in a scrollable div-style container
            st.subheader("Simplified Text")
            st.markdown(
                f'<div style="white-space: pre-wrap; padding:10px; border:1px solid #ccc; border-radius:5px;">{result}</div>',
                unsafe_allow_html=True
            )