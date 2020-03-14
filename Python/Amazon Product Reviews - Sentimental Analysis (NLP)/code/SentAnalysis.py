# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 08:22:40 2019

@author: nitin
"""

# Standard array and dataset libraries
import numpy as np
import pandas as pd

# Model selection, evaluation and data preporcessing libraries
from sklearn.linear_model import LogisticRegression
from xgboost              import XGBClassifier
from sklearn.tree         import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer as tfid
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.corpus import stopwords
from data_util import *

# Miscellaneous Libraries
import time
from pathlib import Path
import random
import pickle


random.seed(20)

start_time = time.time()

# Loading the data
upReviews_file = Path("../data/reviews.pickle")

if not(upReviews_file.is_file()): # If the reviews are being loaded for the first time
    folder_path = '../data/sorted_data/'
    df, X, _, y = load_data(folder_path) 
    
    y = pd.DataFrame(y)
    X = pd.DataFrame(X)
    X['sentiment'] = y
    X = X.rename(columns={0: "review_text"})
    X = X[~(X['sentiment'] == 3)] # Remove reviews with ratings = 3
    
     # Storing the loaded data for future use
    with open("../data/reviews.pickle", "wb") as fp:  
        pickle.dump(X, fp)
else:
    # Load the stored file
    with open("../data/reviews.pickle", "rb") as fp:   # Retrieving the unprocessed reviews
        X = pickle.load(fp)
        print("Unprocessed reviews loaded from stored file")
    
# Sampling the data for faster computation
#X = sample_data(X, 3000)  # If sampling, Move the stored pickle files generated for the whole dataset when the 
                           # script is run for the first time

X_train_up = list(X['review_text'])       
y = list(X['sentiment'])

####################################################

# Data Pre-Processing

prcs_Reviews_file = Path("../data/reviews_processed.pickle")

if not(prcs_Reviews_file.is_file()): # If the reviews are being processed for the first time
    X_clean = preprocess_data(X_train_up)
    
    with open('../data/reviews_processed.pickle', 'wb') as picklefile: # Storing the Processed Reviews
       pickle.dump(X_clean,picklefile)
else:
    with open("../data/reviews_processed.pickle", "rb") as fp:   # Load the Processed Reviews
        X_clean = pickle.load(fp)
        print("\nProcessed Reviews loaded from stored file")

# Load the stopwords provied
stop_words = load_stopwords()

# Remove stopwords manually
#X_clean = removeStopwords(X_clean)

# Converting to bag of words or tfidf
#standard vectorizer   -->  CountVectorizer(binary=True) # Used for DTree and XGBoost Classifier
#n-gram vectorizer     -->  CountVectorizer(binary=True,  ngram_range=(1, 3))
#word-count vectorizer -->  CountVectorizer(binary=False)
#tfid                  -->  tfid(ngram_range=(1, 2), stop_words=stop_words) # Used for LR classifer




X_clean2 = X_clean
vect_data1_file = Path("../data/vectorized_data1.pickle")

if not(vect_data1_file.is_file()): # If the data is being vectorized for the first time
    # Lemmatization
    print("\nLemmatizing the review texts")
    X_clean = lemmatize_reviews(X_clean)
    X_clean2 = X_clean
    
    print("\nVectorizing data to be used for LR and XGBoost Classifier")
    cv = tfid(ngram_range=(1, 2), stop_words=stop_words) # For LR and XGBoost Classifier
    cv.fit(X_clean)
    X_clean = cv.transform(X_clean)
    print("\nData Vectorized for LR and XGBoost Classifier")
    
    with open('../data/vectorized_data1.pickle', 'wb') as picklefile: # Storing the Vectorized Data
       pickle.dump(X_clean,picklefile)
    with open('../data/count_vectorizer_obj1.pickle', 'wb') as pf: # Storing the cv object
       pickle.dump(cv,pf)
else:
    with open("../data/vectorized_data1.pickle", "rb") as fp:   # Load the Vectorized Data
        X_clean = pickle.load(fp)
        print("\nVectorized Data used in LR and XGBoost Classifer loaded from stored file")
    with open("../data/count_vectorizer_obj1.pickle", "rb") as fp2:   # Load the Vectorized Data
        cv = pickle.load(fp2)
        
vect_data2_file = Path("../data/vectorized_data2.pickle")

if not(vect_data2_file.is_file()): # If the data is being vectorized for the first time
    print("\nVectorizing data to be used for DTree Classifier")
    cv_2 = CountVectorizer(binary=True) # For DTree Classifier
    cv_2.fit(X_clean2)
    X_clean2 = cv.transform(X_clean2)
    print("\nData Vectorized for DTree Classifier")
    
    with open('../data/vectorized_data2.pickle', 'wb') as picklefile: # Storing the Vectorized Data
       pickle.dump(X_clean2,picklefile)
    with open('../data/count_vectorizer_obj2.pickle', 'wb') as pf: # Storing the cv object
        pickle.dump(cv_2,pf)
else:
    with open("../data/vectorized_data2.pickle", "rb") as fp:   # Load the Vectorized Data
        X_clean2 = pickle.load(fp)
        print("\nVectorized Data used in DTree Classifer loaded from stored file")
    with open("../data/count_vectorizer_obj2.pickle", "rb") as fp2:   # Load the Vectorized Data
        cv_2 = pickle.load(fp2)


####################################################

# Modeling
X_train, X_val, y_train, y_val = train_test_split(X_clean, y, train_size = 0.75) # For LR and XGBoost
X_train2, X_val2, y_train2, y_val2 = train_test_split(X_clean2, y, train_size = 0.75) # For Dtree
print("\nVectorized Dataset split into Train and Validation set")


# Logistic Regression Clasifier
# Takes about 25-30 minutes to train on the entire review dataset
best_lr_score = 0
best_lr_model = LogisticRegression(C=0.1)
C_vals = [0.01, 0.05, 0.25, 0.5, 1]
lr_cf_file = Path("trained_classifiers/lr_classifer.pickle")

if not(lr_cf_file.is_file()): # If classifying for the first time
    print("\nTraining the LR Classifier")
    for c in C_vals:
        cf = LogisticRegression(C=c)
        cf.fit(X_train, y_train)
        score = accuracy_score(y_val, cf.predict(X_val))
        if(score >= best_lr_score):
            best_lr_model = cf
            best_lr_score = score
        print ("Accuracy for C=%s: %s" % (c, score))

    cf = best_lr_model
    lr_cf = cf
    with open('trained_classifiers/lr_classifer.pickle', 'wb') as picklefile: # Saving the LR classifier
        pickle.dump(cf,picklefile)
else:
    with open('trained_classifiers/lr_classifer.pickle', "rb") as fp:   # Load the LR Classifier
        lr_cf = pickle.load(fp)
        print("\nLogistic Regression Classifier Loaded")
####################################################

   

# XGBoost Classifier
# Takes about 20-25 minutes to train on the entire review dataset
xgb_cf_file = Path("trained_classifiers/xgb_classifer.pickle")

if not(xgb_cf_file.is_file()): # If classifying for the first time
    print("\nTraining the XGB Classifier")
    cf = XGBClassifier()
    cf.fit(X_train, y_train) 
    print("\nXGBoost Classifier trained")
    xgb_cf = cf
    with open('trained_classifiers/xgb_classifer.pickle', 'wb') as picklefile: # Saving the XGBoost classifier
        pickle.dump(cf,picklefile)
else:
    with open('trained_classifiers/xgb_classifer.pickle', "rb") as fp:   # Load the XGBoost Classifier
        xgb_cf = pickle.load(fp)
        print("\nTrained XGBoost Classifier Loaded")
####################################################

# Decision Tree Classifier
# Takes about 2-2.5 hours to train on the entire review dataset
#grid_param_dtree = {'max_depth':[5, 15, 25, 50, 100, 200, 500], 'criterion':['gini', 'entropy'], 'min_samples_split':list(range(2, 150))}
#cf =  GridSearchCV(DecisionTreeClassifier(random_state = 17), grid_param_dtree, cv = 5, scoring  = 'f1', n_jobs = -1)
# Best Params when trained on dataset size of 3000 --> Max Depth = 5, criterion: entropy, min_samples_split = 68
dtree_cf_file = Path("trained_classifiers/dtree_classifer.pickle")

if not(dtree_cf_file.is_file()): # If classifying for the first time
    print("Training the DTree Classifier")
    cf = DecisionTreeClassifier(criterion='entropy', random_state=17)
    cf.fit(X_train2, y_train2)
    print("\nDecision Tree Classifier trained")
    dtree_cf = cf
    with open('trained_classifiers/dtree_classifer.pickle', 'wb') as picklefile: # Saving the DTree classifier
        pickle.dump(cf,picklefile)
else:
    with open('trained_classifiers/dtree_classifer.pickle', "rb") as fp:   # Load the DTree Classifier
        dtree_cf = pickle.load(fp)
        print("\nTrained Decision Tree Classifier Loaded")
####################################################


# Model Evaluation
print("\nModel Evaluation using Logistic Regression Classifier")
y_pred_lr = lr_cf.predict(X_val)
print("Confusion Matrix:")
print(confusion_matrix(y_val,y_pred_lr))
print()
print(classification_report(y_val,y_pred_lr))
print("Accuracy: ",accuracy_score(y_val, y_pred_lr))

print("\nFeatures which are important for determining a positive review:")
feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), lr_cf.coef_[0]
    )
}
for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:5]:
    print()
    print (best_positive)

print("\nFeatures which are important for determining a negative review:")
for best_negative in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:5]:
    print()
    print (best_negative)
    
    
print("\nModel Evaluation using XGBoost Classifier")
y_pred_xgb = xgb_cf.predict(X_val)
print("Confusion Matrix:")
print(confusion_matrix(y_val,y_pred_xgb))
print()
print(classification_report(y_val,y_pred_xgb))
print("Accuracy: ",accuracy_score(y_val, y_pred_xgb))

print("\nModel Evaluation using Decision Tree Classifier")
y_pred_dtree = xgb_cf.predict(X_val2)
print("Confusion Matrix:")
print(confusion_matrix(y_val2,y_pred_dtree))
print()
print(classification_report(y_val2,y_pred_dtree))
print("Accuracy: ",accuracy_score(y_val2, y_pred_dtree))

####################################################


# Total Time taken
print("\nTime Taken:\n--- %s seconds ---" % (time.time() - start_time))