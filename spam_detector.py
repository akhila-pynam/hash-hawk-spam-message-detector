import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

# Fix columns automatically
if 'v1' in data.columns:
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']
else:
    data = data[['label', 'message']]

# Convert labels to numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Remove any missing values
data = data.dropna()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# Convert text to numbers
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Prediction function
def predict(message):
    msg_vec = vectorizer.transform([message])
    result = model.predict(msg_vec)
    return "Spam" if result[0] == 1 else "Not Spam"

# Test
if __name__ == "__main__":
    print(predict("Congratulations! You won a prize"))