# Emotion Classification using Multinomial Naive Bayes

This project builds a simple emotion classification model using text data and the Multinomial Naive Bayes algorithm. The goal is to classify text into one of several emotions such as `joy`, `sadness`, `anger`, etc.

---

## 📂 Dataset Used

- **Name**: `merged_training.pkl`  
- **Format**: Pickle file containing a pandas DataFrame  
- **Columns**:
  - `text`: input text sample
  - `emotions`: corresponding emotion label

---

## 🔍 Approach Summary

1. **Load & preprocess the data** from the pickle file.
2. **Vectorize** the text using `TfidfVectorizer` (max 2000 features).
3. **Split** the data into training and test sets.
4. **Train** a `MultinomialNB` classifier.
5. **Evaluate** using:
   - Accuracy score
   - Confusion matrix

---

## 📦 Dependencies

Install the required libraries using:

```bash
pip install pandas scikit-learn matplotlib
