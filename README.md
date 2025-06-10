import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_pickle("merged_training.pkl")

# 2. Check and clean column names if needed
print("Available columns:", df.columns)
df.columns = df.columns.str.strip()  # Remove accidental whitespace

# 3. Extract features and labels
X = df['text']
y = df['emotions'].str.strip()  # Clean trailing whitespaces or parentheses if any

# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Convert text data into TF-IDF features
vectorizer = TfidfVectorizer(max_features=2000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Train the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 7. Make predictions
y_pred = model.predict(X_test_vec)

# 8. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.4f}")

# 9. Display the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix - Emotion Classification")
plt.tight_layout()
plt.show()
