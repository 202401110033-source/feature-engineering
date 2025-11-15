# Feature Engineering - Fake News Dataset
# Name: Pratik Mane | PRN: 202401110033

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.model_selection import train_test_split

# load files
path = r"D:\Downloads\Compressed\datasets\\"
true = pd.read_csv(path + "true.csv")
fake = pd.read_csv(path + "fake.csv")

# add labels
true["label"] = 0
fake["label"] = 1

# merge datasets
df = pd.concat([true, fake]).reset_index(drop=True)
df = df.fillna("")

print("\n--- Raw Dataset ---")
print(df.head())

# clean text
df["title"] = df["title"].astype(str).str.lower()
df["text"] = df["text"].astype(str).str.lower()
df["combined"] = df["title"] + " " + df["text"]

print("\n--- After Text Cleaning ---")
print(df[["title", "text", "combined"]].head())

# label encode target
le = LabelEncoder()
df["label_enc"] = le.fit_transform(df["label"])

print("\n--- After Label Encoding ---")
print(df[["label", "label_enc"]].head())

# one-hot encode subject
sub_encoded = pd.get_dummies(df["subject"], prefix="sub")

print("\n--- After One-Hot Encoding (subject) ---")
print(sub_encoded.head())

# tf-idf text features
vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
X_tfidf = vectorizer.fit_transform(df["combined"]).toarray()

print("\nTF-IDF Shape:", X_tfidf.shape)

# simple numeric feature
df["text_len"] = df["combined"].apply(len)

print("\n--- Text Length Feature Added ---")
print(df[["combined", "text_len"]].head())

# combine all features
X = np.hstack([
    sub_encoded.values,
    df[["text_len"]].values,
    X_tfidf
])
y = df["label_enc"].values

print("\nFull features:", X.shape)

# remove low variance features
vt = VarianceThreshold(threshold=1e-5)
X_vt = vt.fit_transform(X)

print("After variance threshold:", X_vt.shape)

# chi-square feature selection
k = min(500, X_vt.shape[1])
selector = SelectKBest(chi2, k=k)
X_sel = selector.fit_transform(X_vt, y)

print("After chi-square:", X_sel.shape)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_sel, y, test_size=0.2, random_state=42, stratify=y
)

print("Train:", X_train.shape, "Test:", X_test.shape)
print("\nFeature engineering completed.")


# --- Raw Dataset ---
#                                                title                                               text  ...                date label
# 0  As U.S. budget fight looms, Republicans flip t...  WASHINGTON (Reuters) - The head of a conservat...  ...  December 31, 2017      0
# 1  U.S. military to accept transgender recruits o...  WASHINGTON (Reuters) - Transgender people will...  ...  December 29, 2017      0
# 2  Senior U.S. Republican senator: 'Let Mr. Muell...  WASHINGTON (Reuters) - The special counsel inv...  ...  December 31, 2017      0
# 3  FBI Russia probe helped by Australian diplomat...  WASHINGTON (Reuters) - Trump campaign adviser ...  ...  December 30, 2017      0
# 4  Trump wants Postal Service to charge 'much mor...  SEATTLE/WASHINGTON (Reuters) - President Donal...  ...  December 29, 2017      0

# [5 rows x 5 columns]

# --- After Text Cleaning ---
#                                                title  ...                                           combined
# 0  as u.s. budget fight looms, republicans flip t...  ...  as u.s. budget fight looms, republicans flip t...
# 1  u.s. military to accept transgender recruits o...  ...  u.s. military to accept transgender recruits o...
# 2  senior u.s. republican senator: 'let mr. muell...  ...  senior u.s. republican senator: 'let mr. muell...
# 3  fbi russia probe helped by australian diplomat...  ...  fbi russia probe helped by australian diplomat...
# 4  trump wants postal service to charge 'much mor...  ...  trump wants postal service to charge 'much mor...

# [5 rows x 3 columns]

# --- After Label Encoding ---
#    label  label_enc
# 0      0          0
# 1      0          0
# 2      0          0
# 3      0          0
# 4      0          0

# --- After One-Hot Encoding (subject) ---
#    sub_Government News  sub_Middle-east  sub_News  sub_US_News  sub_left-news  sub_politics  sub_politicsNews  sub_worldnews
# 0                False            False     False        False          False         False              True          False
# 1                False            False     False        False          False         False              True          False
# 2                False            False     False        False          False         False              True          False
# 3                False            False     False        False          False         False              True          False
# 4                False            False     False        False          False         False              True          False

# TF-IDF Shape: (44898, 1000)

# --- Text Length Feature Added ---
#                                             combined  text_len
# 0  as u.s. budget fight looms, republicans flip t...      4724
# 1  u.s. military to accept transgender recruits o...      4142
# 2  senior u.s. republican senator: 'let mr. muell...      2850
# 3  fbi russia probe helped by australian diplomat...      2521
# 4  trump wants postal service to charge 'much mor...      5274

# Full features: (44898, 1009)
# After variance threshold: (44898, 1009)
# After chi-square: (44898, 500)
# Train: (35918, 500) Test: (8980, 500)

# Feature engineering completed.
