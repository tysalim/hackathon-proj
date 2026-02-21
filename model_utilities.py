import pandas as pd
import numpy as np
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
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
# Training Function
# -----------------------------
def train_and_save_model(model_path="model.pkl", tfidf_path="tfidf.pkl"):
    # Load CSV dataset instead of HuggingFace
    df_train = pd.read_csv("data/train.csv").sample(n=20000, random_state=42)
    df_val = pd.read_csv("data/validation.csv")
    df_test = pd.read_csv("data/test.csv")

    df_train = add_readability_features(df_train)
    df_val = add_readability_features(df_val)
    df_test = add_readability_features(df_test)

    df_train["grade_int"] = df_train["grade"].round().astype(int)
    df_val["grade_int"] = df_val["grade"].round().astype(int)
    df_test["grade_int"] = df_test["grade"].round().astype(int)

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=300, ngram_range=(1,2))
    X_train_text = tfidf.fit_transform(df_train["text"])
    X_val_text = tfidf.transform(df_val["text"])
    X_test_text = tfidf.transform(df_test["text"])

    read_train = df_train[FEATURES].fillna(0).values
    read_val = df_val[FEATURES].fillna(0).values
    read_test = df_test[FEATURES].fillna(0).values

    X_train = hstack([X_train_text, read_train])
    X_val = hstack([X_val_text, read_val])
    X_test = hstack([X_test_text, read_test])

    y_train = df_train["grade_int"]
    y_val = df_val["grade_int"]
    y_test = df_test["grade_int"]

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

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Save model + vectorizer
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf, f)

    print("Model saved successfully.")

# -----------------------------
# Load Model
# -----------------------------
def load_model(model_path="model.pkl", tfidf_path="tfidf.pkl"):
    if not os.path.exists(model_path) or not os.path.exists(tfidf_path):
        print("Model not found. Training new model...")
        train_and_save_model(model_path, tfidf_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(tfidf_path, "rb") as f:
        tfidf = pickle.load(f)

    return model, tfidf

# -----------------------------
# Prediction Function
# -----------------------------
def predict_grade_level(text, model, tfidf):
    df_new = pd.DataFrame({"text": [text]})
    df_new = add_readability_features(df_new)

    X_new_text = tfidf.transform(df_new["text"])
    read_new = df_new[FEATURES].fillna(0).values
    X_new = hstack([X_new_text, read_new])

    prediction = np.round(model.predict(X_new))[0].astype(int)
    return prediction