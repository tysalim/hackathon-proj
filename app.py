# app.py
import streamlit as st
import pickle
import numpy as np
from transformers import pipeline
from scipy.sparse import hstack


@st.cache_resource(show_spinner=True)
def load_grade_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=True)
def load_vectorizer():
    with open("vectorizer.pkl", "rb") as f:
        return pickle.load(f)


grade_model = load_grade_model()
vectorizer = load_vectorizer()


@st.cache_resource(show_spinner=True)
def load_simplifier():
    return pipeline("text2text-generation", model="eilamc14/bart-base-text-simplification")

simplifier = load_simplifier()


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Simplify Text", "Grade Level Classifier"])

if page == "Simplify Text":
    st.title("Text Simplifier")
    
    text_input = st.text_area("Enter text to simplify", height=200)
    grade_level = st.slider("Select target reading grade level", 1, 12, 6)
    
    if st.button("Simplify Text"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            # Generate simplified text
            prompt = f"grade {grade_level}: {text_input}"
            result = simplifier(prompt, max_length=512, truncation=True)[0]['generated_text']
            st.subheader("Simplified Text")
            st.write(result)


elif page == "Grade Level Classifier":
    st.title("Reading Grade Level Classifier")
    
    text_input = st.text_area("Enter text to classify", height=200)
    
    if st.button("Classify Grade Level"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            # Preprocess text if needed
            X = vectorizer.transform([text_input])
            predicted_grade = grade_model.predict(X)[0]
            
            st.subheader("Predicted Reading Grade Level")
            st.write(f"Grade {predicted_grade}")