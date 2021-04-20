#%%
import pandas as pd
import numpy as np

data = pd.read_csv('SuicideAndDepression_Detection.csv')

print(data.head())

# Data cleaning:

data.dropna(axis = 0, how ='any', inplace = True)
data.dropna(axis = 1, how ='any', inplace = True)

text = data['text']
labels = data['class']

from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(text, labels, test_size=0.2, random_state=42)

print(len(train_data))
print(len(test_data))

# Making the Count Vectors

#%%
from sklearn.feature_extraction.text import CountVectorizer

counter = CountVectorizer() 
counter.fit(train_data.astype('U').values)
train_counts = counter.transform(train_data.astype('U').values)
test_counts = counter.transform(test_data.astype('U').values)

print(train_data[3])
print(train_counts[3])

# Creating the Naive Bayes Classifier

#%%
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(train_counts, train_labels)
predictions = classifier.predict(test_counts)

# Evaluating the Model
from sklearn.metrics import accuracy_score

print(accuracy_score(test_labels, predictions))

#%%
def transform_predict(input_text):
    text_count = counter.transform([input_text])
    prediction = classifier.predict(text_count)
    prediction_proba = classifier.predict_proba(text_count)

    return prediction, prediction_proba

prediction_proba = transform_predict("I want to kill myself")

prediction_probability = pd.DataFrame(prediction_proba)

# pd.Dataframe(prediction_proba)

print(prediction_probability)

#%%
import plotly.express as px

fig = px.bar(prediction_proba[1])
fig.update_layout(title_text='Bar Graph of Prediction Probability of Each Class')   
fig.show()

# %%
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english')
document_term_matrix = vectorizer.fit_transform(train_data)
vectorizer.get_feature_names()

# %%
print(vectorizer.vocabulary_)

# %%

def get_top_n_words(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    
    get_top_n_words(["I love Python", "Python is a language programming", "Hello world", "I love the world"]) -> 
    [('python', 2),
     ('world', 2),
     ('love', 2),
     ('hello', 1),
     ('is', 1),
     ('programming', 1),
     ('the', 1),
     ('language', 1)]
    """
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# %%
get_top_n_words(train_data, n = 300)

# %%
