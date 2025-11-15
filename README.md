Feature Engineering – Fake News Detection Dataset
📘 Overview

This project applies several feature engineering techniques on a text dataset to prepare it for machine learning.
We use the Fake News Detection Dataset from Kaggle:

🔗 Dataset Link: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

The dataset contains real and fake news articles.
We combine them and extract useful numerical features using text-processing methods.

📂 Dataset Information

We use two CSV files:

Fake.csv → Fake news

True.csv → Real news

A new label column is added:

1 → Fake

0 → True

Columns available:

title

text

subject

date

label (added manually)

Total rows: ~44,898

⚙️ Feature Engineering Steps
1. Data Loading

Load True.csv and Fake.csv

Add labels

Merge into one DataFrame

Print first few rows

2. Data Cleaning

Convert text to lowercase

Combine title + text into combined

Drop empty rows

Print sample rows after cleaning

3. TF-IDF Vectorization

We convert text into numerical values using:

TfidfVectorizer(max_features=1000, stop_words="english")


Produces a matrix of (rows × 1000)

Print shape and sample transformed rows

4. Variance Threshold

Remove features with very low variance:

VarianceThreshold(threshold=0.0001)


Removes uninformative features

Print new shape

5. Chi-Square Feature Selection

Select top 500 most important text features:

SelectKBest(chi2, k=500)


Keeps only meaningful features

Print new shape

6. Train–Test Split

Split final features:

80% Training

20% Testing

Print final shapes.

📊 Output (example)

Your script prints:

Full features: (44898, 1000)
After variance threshold: (44898, XXXX)
After chi-square: (44898, 500)
Train: (35918, 500)
Test: (8980, 500)


Plus sample dataset rows after each step.

🧩 Libraries Used

pandas

numpy

scikit-learn

Install dependencies:

pip install pandas numpy scikit-learn

▶️ How to Run

Download dataset from Kaggle

Place True.csv and Fake.csv in the project folder

Run:

python main.py
