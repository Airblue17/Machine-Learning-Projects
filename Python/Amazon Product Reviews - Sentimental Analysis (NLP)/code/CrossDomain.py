# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 00:25:24 2019

@author: nitin
"""

# Standard array and dataset libraries
import numpy as np
import pandas as pd

# Model selection, evaluation and data preporcessing libraries
from sklearn.tree         import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer as tfid
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from data_util import *

# Miscellaneous Libraries
import time
from pathlib import Path
import random
import pickle

random.seed(20)

folder_path = '../data/sorted_data/'
_, X, product_type, y = load_data(folder_path) 

y = pd.DataFrame(y)
X = pd.DataFrame(X)
prod_type = pd.DataFrame(product_type)
X['sentiment'] = y
X['product_type'] = prod_type
X = X.rename(columns={0: "review_text"})
X = X[~(X['sentiment'] == 3)] # Remove reviews with ratings = 3
train_set = X[X['product_type'] == 'video'] # Loading reviews on videos for train set
test_set = X[X['product_type'] == 'kitchen_&_housewares'] # Loading reviewson kitchen & housewares for test set


X_train_up = list(train_set['review_text'])       
y = list(train_set['sentiment'])

X_test_up = list(test_set['review_text'])       
y_test = list(test_set['sentiment'])

stop_words = load_stopwords()

X_clean = preprocess_data(X_train_up)


print("\nLemmatizing the review texts")
X_clean = lemmatize_reviews(X_clean)

print("\nVectorizing the data")
cv = tfid(max_features = 371490, ngram_range=(1, 2), stop_words=stop_words) 
cv.fit(X_clean)
X_clean = cv.transform(X_clean)
print("\nData Vectorized")


print("Training the model")
cf = DecisionTreeClassifier(criterion='entropy', random_state=17)
cf.fit(X_clean, y)


print("Model Evaluation on the test dataset")
X_test_clean = preprocess_data(X_test_up)



X_test_clean = lemmatize_reviews(X_test_clean)


cv_t = tfid(max_features = 371490, ngram_range=(1, 2), stop_words=stop_words) 
cv_t.fit(X_test_clean)
X_test_clean = cv_t.transform(X_test_clean)

  
y_pred = cf.predict(X_test_clean)
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print()
print(classification_report(y_test,y_pred))
print("Accuracy: ",accuracy_score(y_test, y_pred))   
