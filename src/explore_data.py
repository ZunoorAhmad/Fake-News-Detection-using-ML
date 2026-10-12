
import pandas as pd

# Load both datasets
fake_news = pd.read_csv("data/Fake.csv")
real_news = pd.read_csv("data/True.csv")

# 1. Check dataset sizes
print("FAKE NEWS SHAPE:", fake_news.shape)
print("REAL NEWS SHAPE:", real_news.shape)

# 2. Check column names
print("\nFAKE NEWS COLUMNS:")
print(fake_news.columns.tolist())

print("\nREAL NEWS COLUMNS:")
print(real_news.columns.tolist())

# 3. Check missing values
print("\nMISSING VALUES IN FAKE NEWS:")
print(fake_news.isnull().sum())

print("\nMISSING VALUES IN REAL NEWS:")
print(real_news.isnull().sum())

# 4. Check duplicate rows
print("\nDUPLICATE ROWS IN FAKE NEWS:")
print(fake_news.duplicated().sum())

print("\nDUPLICATE ROWS IN REAL NEWS:")
print(real_news.duplicated().sum())
