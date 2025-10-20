"""
Sentiment Analyzer CLI
- Trains a Logistic Regression on a CSV with columns: text,label (label in {pos,neg})
- Saves/loads model to disk automatically
- Plots confusion matrix after evaluation
Usage:
  python 01_sentiment_cli.py train data.csv
  python 01_sentiment_cli.py eval data.csv
  python 01_sentiment_cli.py predict "I love this!"
Dependencies: scikit-learn, pandas, matplotlib
"""
import sys, os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

MODEL = "sentiment_model.joblib"

def load_df(csv_path):
    df = pd.read_csv(csv_path)
    if not {'text','label'}.issubset(df.columns):
        raise SystemExit("CSV must contain columns: text,label")
    return df

def build_pipe():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=200))
    ])

def train(csv):
    df = load_df(csv)
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])
    pipe = build_pipe().fit(X_train, y_train)
    joblib.dump(pipe, MODEL)
    print(f"Model saved to {MODEL}")
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=sorted(df['label'].unique()))
    fig = plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(range(len(cm)), sorted(df['label'].unique()), rotation=45)
    plt.yticks(range(len(cm)), sorted(df['label'].unique()))
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.show()

def eval(csv):
    if not os.path.exists(MODEL):
        raise SystemExit("Train first.")
    pipe = joblib.load(MODEL)
    df = load_df(csv)
    y_pred = pipe.predict(df['text'])
    print(classification_report(df['label'], y_pred))

def predict(text):
    if not os.path.exists(MODEL):
        raise SystemExit("Train first.")
    pipe = joblib.load(MODEL)
    print(pipe.predict([text])[0])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "train":
        train(sys.argv[2])
    elif cmd == "eval":
        eval(sys.argv[2])
    elif cmd == "predict":
        predict(" ".join(sys.argv[2:]))
    else:
        print(__doc__)
