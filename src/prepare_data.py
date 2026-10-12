
import pandas as pd

# Load the datasets
fake_news = pd.read_csv("data/Fake.csv")
real_news = pd.read_csv("data/True.csv")

# Assign labels
fake_news["label"] = 0
real_news["label"] = 1

# Combine both datasets
news = pd.concat([fake_news, real_news], ignore_index=True)

print("BEFORE CLEANING")
print("Total rows:", len(news))
print("\nClass distribution:")
print(news["label"].value_counts())

# Remove exact duplicate rows
news = news.drop_duplicates()

print("\nAFTER REMOVING EXACT DUPLICATES")
print("Total rows:", len(news))

# Normalize titles and article text for duplicate checking
news["title_check"] = (
    news["title"].fillna("").astype(str).str.strip().str.lower()
)

news["text_check"] = (
    news["text"].fillna("").astype(str).str.strip().str.lower()
)

# Find matching title/text pairs assigned conflicting labels
pairs = news.groupby(["title_check", "text_check"])["label"].nunique()
conflicting_pairs = pairs[pairs > 1]

print("\nTITLE/TEXT PAIRS WITH CONFLICTING LABELS:")
print(len(conflicting_pairs))

# Find repeated title/text pairs, regardless of date or subject
duplicate_pairs = news.duplicated(
    subset=["title_check", "text_check"],
    keep=False
)

print("\nROWS WITH REPEATED TITLE/TEXT PAIRS:")
print(duplicate_pairs.sum())

# Display a few examples for inspection
if duplicate_pairs.any():
    print("\nSAMPLE REPEATED ARTICLES:")
    print(
        news.loc[
            duplicate_pairs,
            ["title", "label", "subject"]
        ].head(10).to_string(index=False)
    )
