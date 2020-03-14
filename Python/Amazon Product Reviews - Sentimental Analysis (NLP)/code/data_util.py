# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 20:58:02 2019

@author: nitin
"""
import numpy as np
import pandas as pd
import os 
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re

def load_data(folder_path):
    '''
    Takes in the folder path of the data
    
    Returns 1. train_data with product_type, unique_id, review_text and rating details
            2. X with review_text details
            3. product_type containing the product information
            4. y with ratings
    '''
    train_data = []
    X = []
    product_type = []
    y = []
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in dirs:
            data_path = folder_path + name + '/all.review' # Read only the all.review file
            print("Loading reviews: "+name)
            with open(data_path, encoding="Latin-1") as file:
               line = file.readline()
               count = 0
               while line:
                   content = line.strip()
                   if(content == "</review>"):
                       count += 1 # Current Review Data End. Increment the index to store the next review value
                       
                   if(content == "<review>"):# Reinitialize variables for a new review
                       uniqueCount = 0
                       storeRev = False
                       storeRat = False
                       product_type.append(name)
                       train_data.append({})
                       train_data[count]["product_type"] = name # Store the product type of the review
            
                   
                   if(uniqueCount == 2):
                       train_data[count]['unique_id'] = content # Store the value of the second unique_id tag of every review
                       uniqueCount = 0           
                   if(content == "<unique_id>"):
                       uniqueCount += 1
                    
                   if(storeRev):
                       train_data[count]['review_text']= content
                       X.append(content)
                       storeRev = False
                   if(content == "<review_text>"):# Next Line will be review Text
                       storeRev = True            # so change storeRev = True
            
                   if(storeRat):
                       rating = int(float(content))
                       if(rating > 3):
                           rating = 1 # Positve
                       elif(rating < 3):
                           rating = 0 # Negative
                       train_data[count]['rating'] = rating
                       y.append(rating)
                       storeRat = False           
                   if(content == "<rating>"): # Next Line will be the rating 
                       storeRat = True        # so change storeRat = True
                   line = file.readline()
    print("\nUnprocessed reviews Loaded")
    return train_data, X, product_type, y

def sample_data(X, n_sample = None):
    
    pos_idx = X['sentiment'].index[X['sentiment'] == 1].tolist()
    neg_idx = X['sentiment'].index[X['sentiment'] == 0].tolist()
    
    if(n_sample == None):
        n_sample = len(neg_idx)
    
    pos_data = X.iloc[pos_idx,:].sample(n = n_sample)
    neg_data = X.iloc[neg_idx,:].sample(n = n_sample)
    
    X = pos_data.append(neg_data).sample(frac = 1)
    
    return X

def load_stopwords():
    stop_words = []
    sw_path = "../data/sorted_data/stopwords"
    with open(sw_path) as file:
        line = file.readline()
        while line:
            content = line.strip()
            stop_words.append(content)
            line = file.readline()
    
    return stop_words

def preprocess_data(data):
    data = [re.compile("[.;:!\'?,\"()\[\]]").sub("", line.lower()) for line in data] # Replace no space
    data = [re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)").sub(" ", line) for line in data] # Replace with space
    print("Data pre-processed")
    return data

def removeStopwords(reviews): # Remove stopwords
    reviews_new = []
    for review in reviews:
        words  = review.split()
        reviews_new.append(
                ' '.join([w for w in words if not w in set(stopwords.words('english'))])
                )
    return reviews_new


def stemming(reviews): # Stemming
    stem = PorterStemmer()
    reviews_new = [' '.join([stem.stem(word) for word in review.split()]) for review in reviews]
    print("Review words stemmed")
    return reviews_new


def lemmatize_reviews(reviews): # Lemmatization
    reviews_new = []
    lem = WordNetLemmatizer()
    reviews_new  =  [' '.join([lem.lemmatize(word) for word in review.split()]) for review in reviews]
    print("Data Lemmatized")
    return reviews_new



