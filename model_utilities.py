import pandas as pd
import numpy as np
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import xgboost as xgb
import pickle
import os

FEATURES = ["fk_grade", "smog_index", "ari", "coleman_liau"]

# -----------------------------
# Feature Engineering
# -----------------------------
def add_readability_features(df):
    df["fk_grade"] = df["text"].apply(textstat.flesch_kincaid_grade)
    df["smog_index"] = df["text"].apply(textstat.smog_index)
    df["ari"] = df["text"].apply(textstat.automated_readability_index)
    df["coleman_liau"] = df["text"].apply(textstat.coleman_liau_index)
    return df

# -----------------------------
# Training & Saving Model
# -----------------------------
def train_and_save_model(model_path="model.pkl", tfidf_path="tfidf.pkl", dataset_csv="readability_data.csv"):
    """
    Train XGBoost model using a CSV dataset and save model + TF-IDF
    """
    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(f"{dataset_csv} not found. Upload your dataset as CSV with 'text' and 'grade' columns.")

    df = pd.read_csv(dataset_csv)
    df = add_readability_features(df)
    df["grade_int"] = df["grade"].round().astype(int)

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=300, ngram_range=(1,2))
    X_text = tfidf.fit_transform(df["text"])
    read_features = df[FEATURES].fillna(0).values
    X = hstack([X_text, read_features])
    y = df["grade_int"]

    # XGBoost Regressor
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        learning_rate=0.1,
        max_depth=4,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mae",
    )
    model.fit(X, y)

    # Save model + vectorizer
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf, f)
    print("Model and TF-IDF vectorizer saved successfully.")

# -----------------------------
# Load Model
# -----------------------------
def load_model(model_path="model.pkl", tfidf_path="tfidf.pkl", dataset_csv="readability_data.csv"):
    if not os.path.exists(model_path) or not os.path.exists(tfidf_path):
        print("Model not found. Training new model...")
        train_and_save_model(model_path, tfidf_path, dataset_csv)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(tfidf_path, "rb") as f:
        tfidf = pickle.load(f)

    return model, tfidf

# -----------------------------
# Predict Function
# -----------------------------
def predict_grade_level(text, model, tfidf):
    df_new = pd.DataFrame({"text": [text]})
    df_new = add_readability_features(df_new)
    X_text = tfidf.transform(df_new["text"])
    read_features = df_new[FEATURES].fillna(0).values
    X = hstack([X_text, read_features])
    prediction = np.round(model.predict(X))[0].astype(int)
    return prediction