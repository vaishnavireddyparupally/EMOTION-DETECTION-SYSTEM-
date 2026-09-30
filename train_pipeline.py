"""
Speech-to-Text Prediction Using NLP — Model Building Pipeline
Follows the project spec: Data Collection -> Understanding -> Feature Engineering ->
Word Cloud -> Preprocessing -> Split -> Vectorization -> Baseline Model -> Evaluation ->
Hyperparameter Tuning -> Best Model Selection -> Save Model
"""

import re
import string
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

# ---------------------------------------------------------------
# Step 2: Data Collection
# ---------------------------------------------------------------
df = pd.read_csv("dataset.csv")
print("Shape:", df.shape)

# ---------------------------------------------------------------
# Step 3: Data Understanding
# ---------------------------------------------------------------
print(df.head())
print(df.info())
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Class distribution:\n", df["target"].value_counts())

# Clean obvious issues found during understanding
df = df.dropna(subset=["text"]).drop_duplicates().reset_index(drop=True)
print("Shape after cleaning:", df.shape)

# ---------------------------------------------------------------
# Step 4: Text Feature Engineering
# ---------------------------------------------------------------
df["num_words"] = df["text"].apply(lambda x: len(str(x).split()))
df["num_chars"] = df["text"].apply(lambda x: len(str(x)))
print(df[["num_words", "num_chars"]].describe())

# ---------------------------------------------------------------
# Step 5: Word Cloud
# ---------------------------------------------------------------
from wordcloud import WordCloud
text_blob = " ".join(df["text"].astype(str))
wc = WordCloud(width=800, height=400, background_color="white").generate(text_blob)
plt.figure(figsize=(10, 5))
plt.imshow(wc)
plt.axis("off")
plt.title("Most Frequent Words in Reviews")
plt.savefig("wordcloud.png", bbox_inches="tight")
plt.close()
print("Saved wordcloud.png")

# ---------------------------------------------------------------
# Step 6: Text Preprocessing
# ---------------------------------------------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(preprocess_text)
print(df[["text", "clean_text"]].head())

# ---------------------------------------------------------------
# Step 7: Input-Output Separation
# ---------------------------------------------------------------
X = df["clean_text"]
y = df["target"]

# ---------------------------------------------------------------
# Step 8: Train-Test Split
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

# ---------------------------------------------------------------
# Step 9: Text Vectorization (TF-IDF)
# ---------------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("Vectorized train shape:", X_train_vec.shape)

# ---------------------------------------------------------------
# Step 10: Baseline Model Building (Logistic Regression)
# ---------------------------------------------------------------
baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train_vec, y_train)

# ---------------------------------------------------------------
# Step 11: Model Evaluation (baseline)
# ---------------------------------------------------------------
y_pred_baseline = baseline_model.predict(X_test_vec)
baseline_acc = accuracy_score(y_test, y_pred_baseline)
print("Baseline Logistic Regression Accuracy:", baseline_acc)
print(classification_report(y_test, y_pred_baseline))

# ---------------------------------------------------------------
# Step 12: Hyperparameter Tuning
# ---------------------------------------------------------------
# Tune Logistic Regression
lr_params = {"C": [0.1, 1, 5, 10], "solver": ["lbfgs"]}
lr_grid = GridSearchCV(LogisticRegression(max_iter=1000), lr_params, cv=5, scoring="accuracy", n_jobs=-1)
lr_grid.fit(X_train_vec, y_train)
print("Best LR params:", lr_grid.best_params_, "CV score:", lr_grid.best_score_)

# Also tune KNN for comparison
knn_params = {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"]}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring="accuracy", n_jobs=-1)
knn_grid.fit(X_train_vec, y_train)
print("Best KNN params:", knn_grid.best_params_, "CV score:", knn_grid.best_score_)

# ---------------------------------------------------------------
# Step 13: Best Model Selection
# ---------------------------------------------------------------
candidates = {
    "LogisticRegression": lr_grid.best_estimator_,
    "KNN": knn_grid.best_estimator_,
}

results = {}
for name, mdl in candidates.items():
    preds = mdl.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    results[name] = {"accuracy": acc, "f1_weighted": f1}
    print(f"{name} -> Accuracy: {acc:.4f}, Weighted F1: {f1:.4f}")

best_name = max(results, key=lambda n: results[n]["f1_weighted"])
best_model = candidates[best_name]
print("Selected best model:", best_name)
print(classification_report(y_test, best_model.predict(X_test_vec)))

# ---------------------------------------------------------------
# Step 14: Save the Model
# ---------------------------------------------------------------
joblib.dump(best_model, "best_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

with open("model_report.txt", "w") as f:
    f.write(f"Best model: {best_name}\n\n")
    for name, r in results.items():
        f.write(f"{name}: accuracy={r['accuracy']:.4f}, weighted_f1={r['f1_weighted']:.4f}\n")
    f.write("\nClassification report (best model):\n")
    f.write(classification_report(y_test, best_model.predict(X_test_vec)))

print("Saved best_model.pkl, vectorizer.pkl, model_report.txt")
