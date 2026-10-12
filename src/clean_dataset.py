
import pandas as pd
from pathlib import Path

# Step 1: Load the original datasets
fake_news = pd.read_csv("data/Fake.csv")
real_news = pd.read_csv("data/True.csv")

print("Original fake news rows:", len(fake_news))
print("Original real news rows:", len(real_news))

# Step 2: Assign labels
# 0 = Fake news, 1 = Real news
fake_news["label"] = 0
real_news["label"] = 1

# Step 3: Combine both datasets
news = pd.concat(
    [fake_news, real_news],
    ignore_index=True
)

print("\nCombined rows:", len(news))

# Step 4: Remove exact duplicate rows
news = news.drop_duplicates().copy()

print("Rows after exact duplicate removal:", len(news))

# Step 5: Normalize title and text for comparison
news["title_check"] = (
    news["title"].fillna("").astype(str).str.strip().str.lower()
)

news["text_check"] = (
    news["text"].fillna("").astype(str).str.strip().str.lower()
)

# Step 6: Remove rows without usable title or article text
news["title"] = news["title"].fillna("").astype(str).str.strip()
news["text"] = news["text"].fillna("").astype(str).str.strip()

news = news[
    (news["title"] != "") &
    (news["text"] != "")
].copy()

# Step 7: Find identical title/text pairs with conflicting labels
pair_labels = news.groupby(
    ["title_check", "text_check"]
)["label"].nunique()

conflicting_pairs = pair_labels[pair_labels > 1].index

conflict_mask = pd.MultiIndex.from_frame(
    news[["title_check", "text_check"]]
).isin(conflicting_pairs)

print("\nConflicting article pairs:", len(conflicting_pairs))
print("Rows involved in conflicts:", int(conflict_mask.sum()))

# Exclude conflicting articles from this experiment
news = news.loc[~conflict_mask].copy()

# Step 8: Remove temporary comparison columns
news = news.drop(
    columns=["title_check", "text_check"]
)

# Step 9: Combine headline and body into one text feature
news["content"] = news["title"] + " " + news["text"]

# Step 10: Inspect the final dataset
print("\nFinal dataset shape:", news.shape)

print("\nFinal label distribution:")
print(
    news["label"]
    .value_counts()
    .sort_index()
    .rename(index={0: "Fake", 1: "Real"})
)

# Step 11: Save the cleaned dataset
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "news_cleaned.csv"

news.to_csv(output_file, index=False)

print("\nCleaned dataset saved to:", output_file)
