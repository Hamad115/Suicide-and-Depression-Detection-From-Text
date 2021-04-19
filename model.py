# Model:
import pandas as pd
import numpy as np

data = pd.read_csv('SuicideAndDepression_Detection.csv')

data.dropna(axis = 0, how ='any', inplace = True)
data.dropna(axis = 1, how ='any', inplace = True)

text = data['text']
labels = data['class']

from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(text, labels, test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer

counter = CountVectorizer() 
counter.fit(train_data.astype('U').values)
train_counts = counter.transform(train_data.astype('U').values)
test_counts = counter.transform(test_data.astype('U').values)

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(train_counts, train_labels)

def transform_predict(input_text):
    text_count = counter.transform([input_text])
    prediction = classifier.predict(text_count)
    prediction_proba = classifier.predict_proba(text_count)

    return prediction, prediction_proba

