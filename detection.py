import pandas as pd
import numpy as np

data = pd.read_csv('SuicideAndDepression_Detection.csv')

print(data.head())

# Data cleaning:

data.isnull().sum()

data.dropna(axis = 0, how ='any', inplace = True)
data.dropna(axis = 1, how ='any', inplace = True)

text = data['text']
labels = data['class']

data.info()

from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(text, labels, test_size=0.2, random_state=42)

print(len(train_data))
print(len(test_data))

# Making the Count Vectors

from sklearn.feature_extraction.text import CountVectorizer

counter = CountVectorizer() 
counter.fit(train_data.astype('U').values)
train_counts = counter.transform(train_data.astype('U').values)
test_counts = counter.transform(test_data.astype('U').values)

print(train_data[3])
print(train_counts[3])

# Creating the Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(train_counts, train_labels)
predictions = classifier.predict(test_counts)

# Evaluating the Model
from sklearn.metrics import accuracy_score

print(accuracy_score(test_labels, predictions))

def transform_predict(input_text):
    text_count = counter.transform([input_text])
    prediction = classifier.predict(text_count)
    prediction_proba = classifier.predict_proba(text_count)

    return prediction, prediction_proba

prediction_proba = transform_predict("I feel lonely and unwanted")

prediction_probability = pd.DataFrame(prediction_proba)

print(prediction_probability)