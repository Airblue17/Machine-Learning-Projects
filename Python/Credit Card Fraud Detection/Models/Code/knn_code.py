# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 17:42:34 2019

@author: nitin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import metrics, preprocessing as prep
from sklearn.decomposition import PCA
import random

def knn_part():
    print("\nQuestion 2 (KNN)")
    #loading the data
    train = pd.read_csv("../../Data/x_train.csv", header = None)
    y_train = pd.read_csv("../../Data/y_train.csv", header = None)
    
    #checking the data
    #print(train.head)
    #print(train.describe())
    random.seed(99)
    #Feature normalization
    train_normalized = pd.DataFrame(prep.normalize(train)) 
    
    #The dataset is skewed and has a high amount of non-fraudulent transactions
    #To make sure the model is not biased towards the non-fraudulent features more
    #Sampling the dataset to have equal number of fraud and non fraud examples
    num_fraud = np.sum(y_train[0] == 1) #print(np.sum(y_train[0] == 1))
    
    #print(np.sum(y_train[0] == 0))
    
    #indexes where labels are fraud/non_fraud
    fraud_idx = y_train.index[y_train[0] == 1].tolist()
    non_fraud_idx =  y_train.index[y_train[0] == 0].tolist()
    
    #creating a new temporary fraud data frame
    train_fr = train_normalized.iloc[fraud_idx,:]
    y_train_fr = y_train.iloc[fraud_idx,:]
    train_fr = train_fr.assign(labels = y_train_fr)
    
    #creating a new temporary non fraud data frame
    train_nfr = train_normalized.iloc[non_fraud_idx,:]
    y_train_nfr = y_train.iloc[non_fraud_idx,:]
    #randomly sampling num_fraud examples from not fraud data frame
    train_nfr = train_nfr.assign(labels = y_train_nfr).sample(n = num_fraud, random_state = 17)
    
    #Shuffling the rows in the new train data set with equal fraud and non fraud examples
    train_new = train_fr.append(train_nfr).sample(frac = 1, random_state = 17)
    y_train_new = train_new['labels']
    del train_new['labels']
    
    pca = PCA().fit(train_new)#Plotting the Cumulative Summation of the Explained Variance
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') #for each component
    plt.title('Pulsar Dataset Explained Variance')
    plt.xticks(np.arange(0,40,2))
    plt.show()
    
    pca = PCA(n_components=23)
    train_new = pd.DataFrame(pca.fit_transform(train_new))
    
    #Implementing cross validation
    num_folds = 5
    k_values = [3, 5, 10, 20, 25]
    best_f_score = 0
    best_k = -1
    
    
    #Splitting the training data into folds
    train_folds = np.array_split(train_new, num_folds)
    y_train_folds = np.array_split(y_train_new, num_folds)
    final_f_scores = {}
    for k in k_values:
        best_f_score_k = 0
        f_score_avg_k = 0
        print("\nK is: %d" % (k))
        for itr in range(num_folds):
            #model definition
            model = knn(n_neighbors = k)
            
            #Getting the validation fold as a pandas data frame
            val_fold = pd.DataFrame(train_folds[itr])
            val_labels = pd.DataFrame(y_train_folds[itr])
            
            #Remaining folds are assinged to the train fold
            train_fold = pd.DataFrame(np.concatenate([fold for idx, fold in enumerate(train_folds) if idx != itr]))
            train_labels = pd.DataFrame(np.concatenate([fold for idx, fold in enumerate(y_train_folds) if idx != itr]))
            
            #Training the model
            model.fit(train_fold, train_labels.values.ravel())
            
            #PRediction
            y_val_pred = model.predict(val_fold)
            f1_score_val = metrics.f1_score(val_labels, y_val_pred, average = 'binary', pos_label = 0)
            print(f1_score_val)
            f_score_avg_k += f1_score_val
            #Update the best accuracy for current k 
            if(f1_score_val > best_f_score_k):
                best_f_score_k = f1_score_val        
        print("For k = %d, Max F-Score: %f" % (k, best_f_score_k))
        #Computing the average of fscores for a particular k
        f_score_avg_k /= 5
        #storing the avg f score for k in a dictionary
        final_f_scores[k] = f_score_avg_k
        #update the best accuracy overall
        if(f_score_avg_k > best_f_score):
           best_f_score =  f_score_avg_k
           best_k = k
           
    print("\nThe Best F-score average: %f achieved when k = %d" % (best_f_score, best_k))
    print("\nAverage F-scores for different values of k after cross-validation:")
    print(final_f_scores)
    print("\nQuestion 2 End")
