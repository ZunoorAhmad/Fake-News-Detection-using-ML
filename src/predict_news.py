
import joblib
from pathlib import Path

# Locate the trained model
model_path = Path("models/fake_news_model.joblib")

if not model_path.exists():
    raise FileNotFoundError(
        "Trained model not found. Please run src/train_model.py first."
    )

# Load the trained model
model = joblib.load(model_path)

print("Fake News Detector")
print("Enter a news article to classify.")
print("Type 'exit' to quit.\n")

while True:
    news_text = input("Enter news text: ").strip()

    if news_text.lower() == "exit":
        print("Goodbye!")
        break

    if not news_text:
        print("Please enter some news text.\n")
        continue

    # Make a prediction
    prediction = model.predict([news_text])[0]

    # Convert numeric label into readable text
    label = "Fake" if prediction == 0 else "Real"

    # Get prediction probabilities
    probabilities = model.predict_proba([news_text])[0]
    confidence = probabilities[int(prediction)] * 100

    print(f"Prediction: {label}")
    print(f"Model confidence: {confidence:.2f}%\n")
