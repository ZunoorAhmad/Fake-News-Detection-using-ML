
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Step 1: Load the cleaned dataset
input_file = Path("data/processed/news_cleaned.csv")
news = pd.read_csv(input_file)

print("Loaded dataset:", news.shape)

# Step 2: Remove repeated article content
# Keep one copy of each exactly matching content string
before = len(news)

news = news.drop_duplicates(
    subset=["content"],
    keep="first"
).copy()

print("Repeated content rows removed:", before - len(news))
print("Rows after content deduplication:", len(news))

# Step 3: Select model input and target label
X = news["content"]
y = news["label"]

# Step 4: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Step 5: Save the splits
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

train_data = pd.DataFrame({
    "content": X_train,
    "label": y_train
})

test_data = pd.DataFrame({
    "content": X_test,
    "label": y_test
})

train_data.to_csv(output_dir / "train.csv", index=False)
test_data.to_csv(output_dir / "test.csv", index=False)

# Step 6: Display the results
print("\nTRAINING DATA")
print("Rows:", len(X_train))
print("Class distribution:")
print(y_train.value_counts().sort_index())

print("\nTESTING DATA")
print("Rows:", len(X_test))
print("Class distribution:")
print(y_test.value_counts().sort_index())

print("\nTraining and testing files saved successfully.")
