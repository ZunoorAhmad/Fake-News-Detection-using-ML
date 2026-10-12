
import pandas as pd
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Step 1: Load the prepared training and testing data
train_data = pd.read_csv("data/processed/train.csv")
test_data = pd.read_csv("data/processed/test.csv")

X_train = train_data["content"].fillna("")
y_train = train_data["label"]

X_test = test_data["content"].fillna("")
y_test = test_data["label"]

print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))

# Step 2: Create the machine learning pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

# Step 3: Train the model
print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")

# Step 4: Predict labels for unseen test examples
y_pred = model.predict(X_test)

# Step 5: Evaluate performance
print("\nAccuracy:")
print(round(accuracy_score(y_test, y_pred), 4))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Fake", "Real"],
        digits=4
    )
)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 6: Save the trained model
output_dir = Path("models")
output_dir.mkdir(parents=True, exist_ok=True)

model_path = output_dir / "fake_news_model.joblib"
joblib.dump(model, model_path)

print("\nTrained model saved to:", model_path)
