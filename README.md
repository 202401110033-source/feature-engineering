# feature-engineering
#https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

## 📘 Overview
This project demonstrates various **Feature Engineering techniques** applied to a text-based dataset.  
The chosen dataset is the **Fake News Detection Dataset** from Kaggle, containing real and fake news articles.  
The goal is to transform raw textual data into informative numerical features suitable for machine learning models.

---

## 📂 Dataset Information
**Dataset:** Fake News Detection (Kaggle)  
**Total Records:** 44,898  
**Columns:**
- `title` — Headline of the article  
- `text` — Full content of the news  
- `subject` — Category or topic of the article  
- `date` — Date of publication  
- `label` — Target variable (1 = Fake, 0 = True)

The dataset was combined from two CSV files: `True.csv` and `Fake.csv`.

---

## ⚙️ Steps Performed

### 1. Data Loading & Exploration
- Loaded both CSV files (`True.csv`, `Fake.csv`).
- Added target labels (0 for True, 1 for Fake).
- Checked data structure and verified no missing values.

### 2. Data Cleaning
- Converted text to lowercase.
- Combined `title` and `text` columns into a single column called `combined` for richer meaning.

### 3. Text Encoding
- Used **TF-IDF Vectorization** to convert text into numerical format.
- Limited vocabulary size to 1000 features for efficiency.

### 4. Feature Scaling
- Applied **StandardScaler** to standardize all numeric features (mean = 0, std = 1).

### 5. Feature Extraction
- Performed **Principal Component Analysis (PCA)** to reduce dimensions from 1000 → 50 components.

### 6. Feature Selection
- Used **Variance Threshold** to remove low-variance (uninformative) features.

### 7. Final Dataset
- The final dataset consists of 50 meaningful components.
- Split into 80% training and 20% testing data.

---

## 🧩 Libraries Used
- `pandas` — Data handling  
- `numpy` — Numerical operations  
- `scikit-learn` — TF-IDF, PCA, scaling, and feature selection  
- `matplotlib`, `seaborn` — Visualization  

To install all dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
