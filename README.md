Feature Engineering – Fake News Detection Dataset

This project demonstrates basic feature engineering techniques on the Fake News Detection dataset from Kaggle.
The goal is to convert raw news text into meaningful numerical features suitable for machine learning.
Dataset link: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

Dataset
Two CSV files are used: True.csv and Fake.csv.
Labels are assigned as:
0 – True (real news)
1 – Fake

Columns included: title, text, subject, date (and label added).
Total records: around 44,898.

Steps Performed

Data Loading & Merging
– Loaded True.csv and Fake.csv.
– Added labels.
– Combined both files into one dataset.

Data Cleaning
– Filled missing values with blank string.
– Converted all text to lowercase.
– Created a new column “combined” by merging title and text for better meaning.

Label Encoding
– Encoded the target label into numeric form using LabelEncoder.

One-Hot Encoding
– Subject column was transformed using one-hot encoding so each subject becomes a separate binary column.

TF-IDF Feature Extraction
– Applied TF-IDF vectorizer to the “combined” text.
– Limited vocabulary to 1000 features for performance and memory efficiency.

Simple Numeric Feature
– Added a new column “text_len” that stores the length of the combined text.

Combining All Features
– Used NumPy to merge one-hot encoded features, text_len, and TF-IDF vectors into a single feature matrix.

Variance Threshold
– Removed features with extremely low variance using VarianceThreshold.
– Helps remove columns that do not contribute meaningful information.

Chi-Square Feature Selection
– Selected the top 500 most important features using chi-square test.
– Reduces dimensionality while keeping important text-related features.

Train-Test Split
– Split the data into 80% training and 20% testing.

Example Output Printed by the Code
– First few rows of raw dataset
– First few rows after text cleaning
– Subject one-hot encoding preview
– Shapes after TF-IDF
– Shapes after feature selection
– Train and test shapes

Final shapes expected:
Full features: (44898, 1009)
After variance threshold: (44898, 1009)
After chi-square: (44898, 500)
Train: (35918, 500)
Test: (8980, 500)

Libraries Used
pandas
numpy
scikit-learn

How to Run

Download Fake.csv and True.csv from Kaggle.

Place them in the same folder as the Python script.

Update the path in the script as needed.

Run the program using: python main.py
You will see dataset examples and transformation steps printed in the console.

Ethics
This dataset contains only text-based news articles and does not include personal attributes such as gender, age, or marital status.
Models built on such text data are free from demographic bias.
If using personal data in the future, sensitive features must be removed or anonymized to avoid discrimination.
