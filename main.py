'''
This script performs textual sentiment anaysis on movie revies posted by users on IMDB.
The dataset can be found at https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
'''

#Module importing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#Creating pandas dataframe of the dataset
data = pd.read_csv('IMDB Dataset.csv')
print(data.head())

# Review Length Analysis
data['sentiment'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

#Preprocessing data
data['review'] = data['review'].str.lower()
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# using train_test_split to test fit
X = data['review']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# vectorisation of data
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# fitting the data onto Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# model prediction data
y_pred = model.predict(X_test_vec)

# accuracy metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# printing accuracy scores
print()
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)
print("Confusion Matrix:")
print(cm)

# visualising confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()