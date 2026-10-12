# Fake News Detection Using Machine Learning

## Project Overview

This is a beginner-friendly machine learning project focused on classifying news articles as fake or real using Python and Natural Language Processing (NLP).

The purpose of this project is to learn the complete machine learning workflow, starting with dataset exploration and preparation, followed by model training, evaluation, and prediction.

This project is currently focused only on machine learning. It does not include a frontend, backend API, or web application.

## Technologies Used

- Python
- pandas
- scikit-learn
- Jupyter
- matplotlib

## Project Structure

```text
Fake News Check/
├── data/
│   ├── Fake.csv
│   ├── True.csv
│   └── processed/
├── notebooks/
├── src/
│   ├── explore_data.py
│   ├── prepare_data.py
│   └── clean_dataset.py
├── .venv/
├── requirements.txt
└── README.md
```

## Learning Progress

### Step 1: Environment Setup

Created a Python virtual environment, installed the required libraries, and established the project directory structure.

### Step 2: Dataset Exploration

**File:** `src/explore_data.py`

Loaded the fake and real news CSV files using pandas and inspected their shapes, columns, missing values, and duplicate rows.

Initial findings:

- Fake news: 23,481 rows.
- Real news: 21,417 rows.
- Both files contain four columns: `title`, `text`, `subject`, and `date`.
- No missing values were detected in the initial inspection.
- Three exact duplicate rows were found in the fake news file.
- 206 exact duplicate rows were found in the real news file.

### Step 3: Initial Dataset Preparation Experiments

**File:** `src/prepare_data.py`

Created an initial script to combine both datasets, assign numerical labels, inspect class distribution, and investigate duplicate articles and potentially conflicting labels.

Labels:

- `0` = Fake news.
- `1` = Real news.

This script is preserved as a record of the initial data-preparation experiment.

### Step 4: Dataset Cleaning

**File:** `src/clean_dataset.py`

Created a separate cleaning script without overwriting earlier work.

The script combines the datasets, removes exact duplicates, identifies matching title/body pairs with conflicting labels, removes unusable text rows, creates a combined `content` column, and saves the prepared dataset to:

`data/processed/news_cleaned.csv`

The cleaning results will be recorded after running the script.

## Important Notes

The dataset labels are inherited from the source dataset and should not be considered independently verified facts.

Removing duplicates and conflicting labels helps improve data consistency, but further checks are required to prevent data leakage and assess whether the model generalizes to unseen news.

### Step 5: Training and Testing Split

**File:** `src/split_dataset.py`

Loaded the cleaned dataset and removed repeated article content before splitting the data.

Used scikit-learn's `train_test_split` to divide the dataset into:

- 80% training data.
- 20% testing data.

Used `stratify=y` to maintain approximately consistent class proportions and `random_state=42` to make the split reproducible.

The script saves:

- `data/processed/train.csv`
- `data/processed/test.csv`

The training data will be used to fit the model. The testing data will be held aside until evaluation.

The initial split is a learning baseline. Further checks for near-duplicate articles and source-related data leakage may be needed to estimate real-world performance reliably.

### Step 6: First Machine Learning Model

**File:** `src/train_model.py`

Trained an initial news-text classification model using a scikit-learn pipeline containing:

- TF-IDF for converting text into numerical features.
- Logistic Regression for binary classification.

The model was trained using `data/processed/train.csv` and evaluated using the held-out `data/processed/test.csv`.

Evaluation metrics recorded:

- Accuracy: [Add actual result]
- Fake-class precision: [Add actual result]
- Fake-class recall: [Add actual result]
- Fake-class F1 score: [Add actual result]
- Real-class precision: [Add actual result]
- Real-class recall: [Add actual result]
- Real-class F1 score: [Add actual result]

The fitted pipeline is saved to `models/fake_news_model.joblib`.

These metrics represent performance on this dataset split and should not be interpreted as proof that the model can reliably determine whether arbitrary real-world news is true or false.

## Step 6: Train the Machine Learning Model

### Objective
Train a machine learning model to classify news articles as Fake or Real using their text.

### Algorithms and Libraries
- **TF-IDF Vectorizer:** Converts text into numerical features that a machine learning model can process.
- **Logistic Regression:** Learns patterns from labeled examples to predict whether an article is Fake or Real.
- **Scikit-learn:** Provides the text-processing pipeline, model training, and evaluation metrics.
- **Joblib:** Saves the trained model for future use.

### Training Process
1. Loaded the training and testing datasets.
2. Converted article text into TF-IDF features using unigrams and bigrams.
3. Trained a Logistic Regression classifier.
4. Evaluated predictions against the test labels.
5. Saved the trained pipeline.

### Results

| Metric | Result |
|---|---:|
| Training examples | 30,924 |
| Testing examples | 7,732 |
| Accuracy | 98.41% |
| Fake precision | 99.07% |
| Fake recall | 97.39% |
| Fake F1-score | 98.22% |
| Real precision | 97.88% |
| Real recall | 99.25% |
| Real F1-score | 98.56% |

### Confusion Matrix

| Actual / Predicted | Fake | Real |
|---|---:|---:|
| Fake | 3,402 | 91 |
| Real | 32 | 4,207 |

### Saved Model
The trained model is saved at:

`models/fake_news_model.joblib`

The saved file contains the TF-IDF vectorizer and Logistic Regression classifier together in a single pipeline.

### Important Limitation
The model predicts patterns associated with the dataset's labels. Its predictions are not definitive proof that a news article is true or false. Dataset bias, misleading labels, and changes in writing styles can affect real-world performance.

### Next Step
Create a separate prediction script that loads the saved model and classifies new text without retraining it.

## Step 7: Predict New News Articles

### Objective
Use the previously trained model to classify new news text without retraining it.

### Implementation
Created `src/predict_news.py`, which:
1. Loads the saved model from `models/fake_news_model.joblib`.
2. Accepts news text through terminal input.
3. Predicts the class using the trained pipeline.
4. Displays the prediction as Fake or Real.
5. Displays the model's estimated probability for the predicted class.
6. Allows the user to continue testing until they type `exit`.

### How to Run
```bash
python src/predict_news.py
```

### Important Limitation
The predicted label and confidence score are based on patterns learned from the training dataset. They do not establish whether a news story is factually correct. Real-world reliability requires further testing on independent and diverse news sources.

## Next Steps

1. Verify the cleaned dataset.
2. Learn how training and testing datasets work.
3. Convert news text into numerical features using TF-IDF.
4. Train a baseline Logistic Regression classifier.
5. Evaluate the model using precision, recall, F1 score, and a confusion matrix.
6. Test predictions on previously unseen text.

The goal is to build an understandable, reproducible machine learning experiment and document each learning stage.
# Fake-News-Detection-using-ML
