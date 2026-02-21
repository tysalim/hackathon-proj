import streamlit as st
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
from model_utilities import load_model, predict_grade_level
import torch


# Page Config
st.set_page_config(
    page_title="Readability Toolkit",
    layout="wide"
)


# Cached Resources
@st.cache_resource(show_spinner=True)
def get_grade_model():
    return load_model()

@st.cache_resource(show_spinner=True)
def get_simplifier():
    model_name = "eilamc14/bart-base-text-simplification"

    # Attempt to load text2text-generation pipeline
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
        # Fallback for older Transformers versions
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)

        def simplifier(text, max_length=512):
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=max_length)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.session_state["pipeline_type"] = "fallback"
        return simplifier

# Load models
model, tfidf = get_grade_model()
simplifier = get_simplifier()


# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a tool:",
    ["Grade Level Classifier", "Text Simplifier"]
)


# PAGE 1 — GRADE CLASSIFIER
if page == "Grade Level Classifier":
    st.title("Reading Grade Level Classifier")

    text_input = st.text_area(
        "Enter text to classify:",
        height=200
    )

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


# PAGE 2 — TEXT SIMPLIFIER 
elif page == "Text Simplifier":
    st.title("Text Simplifier")

    text_input = st.text_area(
        "Enter text to simplify:",
        height=200
    )

    target_grade = st.slider(
        "Select target reading grade level:",
        min_value=1,
        max_value=12,
        value=6
    )

    def clean_output(text):
        """
        Clean and format model output:
        - Remove repeated words/tokens
        - Remove extra spaces
        - Ensure proper sentence spacing
        """
        import re

        # Remove repeated words (e.g., "the the" -> "the")
        text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

        # Replace multiple punctuation with single
        text = re.sub(r'([.!?])\1+', r'\1', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Capitalize first letter after sentence end
        sentences = re.split(r'([.!?])', text)
        cleaned = ""
        for i in range(0, len(sentences)-1, 2):
            s = sentences[i].strip()
            p = sentences[i+1]
            if s:
                s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
                cleaned += s + p + " "
        cleaned = cleaned.strip()
        return cleaned

    if st.button("Simplify Text"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Simplifying text..."):
                # Stronger prompt to force simplification
                prompt = (
                    f"Simplify this text for a young student (grade {target_grade}):\n"
                    "- Use short sentences (≤10 words each)\n"
                    "- Use only common, simple words\n"
                    "- Replace difficult words with simpler alternatives\n"
                    f"Text: {text_input}"
                )

                # Run through the pipeline
                if st.session_state.get("pipeline_type") == "text2text":
                    raw_output = simplifier(prompt, max_length=512, truncation=True)[0]["generated_text"]
                else:
                    raw_output = simplifier(prompt, max_length=512)

                # Clean and format the output
                result = clean_output(raw_output)

            st.subheader("Simplified Text")
            st.write(result)