# Name:-Pratik Mane
#PRN:-202401110033
# =============================================
# Feature Engineering Assignment
# Dataset: Fake News Detection (True.csv + Fake.csv)
# =============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2

path = r"C:\Users\prati\Downloads\Compressed\datasets\\"

true = pd.read_csv(path + "true.csv")
fake = pd.read_csv(path + "fake.csv")

true["label"] = 0
fake["label"] = 1

df = pd.concat([true, fake])
df = df.sample(frac=1).reset_index(drop=True)  # shuffle
print("✅ Dataset Loaded Successfully!\n")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

df = df.fillna("")

df["text"] = df["text"].str.lower()
df["title"] = df["title"].str.lower()

df["combined"] = df["title"] + " " + df["text"]

vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
X = vectorizer.fit_transform(df["combined"])
y = df["label"]

scaler = StandardScaler(with_mean=False)  # sparse matrix compatible
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled.toarray())
print("\n✅ PCA applied. Reduced from", X_scaled.shape[1], "→", X_pca.shape[1])

selector = VarianceThreshold(threshold=0.0001)
X_selected = selector.fit_transform(X_pca)
print("✅ Variance Threshold applied. Features left:", X_selected.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

print("\n✅ Train/Test split done.")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\n🎯 Summary:")
print("- Combined title + text for richer features")
print("- Applied TF-IDF (1000 features)")
print("- Standardized features")
print("- Applied PCA (50 components)")
print("- Applied Variance Threshold for feature selection")
print("- Ready for model training!")

plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Explained Variance Ratio')
plt.grid(True)
plt.tight_layout()
plt.show()

#Output

# ✅ Dataset Loaded Successfully!

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 44898 entries, 0 to 44897
# Data columns (total 5 columns):
#  #   Column   Non-Null Count  Dtype 
# ---  ------   --------------  ----- 
#  0   title    44898 non-null  object
#  1   text     44898 non-null  object
#  2   subject  44898 non-null  object
#  3   date     44898 non-null  object
#  4   label    44898 non-null  int64 
# dtypes: int64(1), object(4)
# memory usage: 1.7+ MB
# None

# Missing values:
#  title      0
# text       0
# subject    0
# date       0
# label      0
# dtype: int64

# ✅ PCA applied. Reduced from 1000 → 50
# ✅ Variance Threshold applied. Features left: 50

# ✅ Train/Test split done.
# Train shape: (35918, 50)
# Test shape: (8980, 50)

# 🎯 Summary:
# - Combined title + text for richer features
# - Applied TF-IDF (1000 features)
# - Standardized features
# - Applied PCA (50 components)
# - Applied Variance Threshold for feature selection
# - Ready for model training!