# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 22:44:21 2019

@author: nitin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier  as dtree
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, preprocessing as prep
import random
import time
from kaggle import *
from sklearn.model_selection  import GridSearchCV

def decision_tree_part():
    print("\nQuestion 3(Decision Tree)")
    #loading the data
    train = pd.read_csv("../../Data/x_train.csv", header = None)
    y_train = pd.read_csv("../../Data/y_train.csv", header = None)
    
    random.seed(99)
    
    #print(train.describe())
            
    #The dataset is skewed and has a high amount of non-fraudlent transactions
    #To make sure the model is not biased towards the non-fraudulent features more
    #Sampling the dataset to have equal number of fraud and non fraud examples
    num_fraud = np.sum(y_train[0] == 1) #print(np.sum(y_train[0] == 1))
    
    #print(np.sum(y_train[0] == 0))
    
    #indexes where labels are fraud/non_fraud
    fraud_idx = y_train.index[y_train[0] == 1].tolist()
    non_fraud_idx =  y_train.index[y_train[0] == 0].tolist()
    
    #creating a new temporary fraud data frame
    train_fr = train.iloc[fraud_idx,:]
    y_train_fr = y_train.iloc[fraud_idx,:]
    train_fr = train_fr.assign(labels = y_train_fr)
    
    #creating a new temporary non fraud data frame
    train_nfr = train.iloc[non_fraud_idx,:]
    y_train_nfr = y_train.iloc[non_fraud_idx,:]
    #randomly sampling 2 * num_fraud examples from not fraud data frame
    train_nfr = train_nfr.assign(labels = y_train_nfr).sample(n = num_fraud * 2, random_state = 17)
    
    #Shuffling the rows in the new train data set with equal fraud and non fraud examples
    train_new = train_fr.append(train_nfr).sample(frac = 1, random_state = 17)
    y_train_new = train_new['labels']
    del train_new['labels']
    
    #Implementing cross validation
    num_folds = 5
    depth_values = [3, 6, 9, 12, 15]
    best_f_score = 0
    best_depth = -1
    model = dtree()
    
    time_measured = {}
    #Splitting the training data into folds
    train_folds = np.array_split(train_new, num_folds)
    y_train_folds = np.array_split(y_train_new, num_folds)
    final_f_scores = {}
    for depth in depth_values:
        best_f_score_depth = 0
        f_score_avg_depth = 0
        print("\nmax depth is: %d" % (depth))
        start = datetime.datetime.now() #for measuring time taken for cross-validation 
        for itr in range(num_folds):
            #model definition
            model = dtree(max_depth = depth, criterion = "entropy", random_state = 17)
            
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
            f1_score_val = metrics.f1_score(val_labels, y_val_pred)
            print(f1_score_val)
            f_score_avg_depth += f1_score_val
            #Update the best accuracy for current depth 
            if(f1_score_val > best_f_score_depth):
                best_f_score_depth = f1_score_val         
        print("For max depth = %d, Max F-Score: %f" % (depth, best_f_score_depth))
        #Computing the average of fscores for a particular depth
        f_score_avg_depth /= 5
        #storing the avg f score for depth in a dictionary
        final_f_scores[depth] = f_score_avg_depth
        #update the best accuracy overall
        if(f_score_avg_depth > best_f_score):
           best_model = model
           best_f_score =  f_score_avg_depth
           best_depth = depth
        end = datetime.datetime.now()
        time_measured[depth] = int((end-start).total_seconds()*1000)
    print("\nThe Best F-score Average: %f achieved when depth = %d" % (best_f_score, best_depth))
    print("\nMax depth of the tree and corresponding average F-scores after cross-validation:")
    print(final_f_scores)
    
    
    #Plotting graph for time taken to perform cross validation
    x = [key for (key, value) in time_measured.items()]
    y = [value for (key, value) in time_measured.items()]
    fig, ax = plt.subplots(1)
    plt.plot(x, y)
    plt.xticks(np.arange(3,18,3))
    ax.set_ylim(bottom=np.min(y)-2, top = np.max(y)+2)
    plt.yticks(np.arange(np.min(y) - 2, np.max(y) + 2, 2))
    plt.xlabel("Depth Values")
    plt.ylabel("Time taken to perform cross-validation (in ms)")
    plt.title("Graph for depth values vs. Time taken to perform cross-validation")
    plt.show(fig)
    fig.savefig("../Figures/depthVStimetaken.png")
    
    #########Question 3.3 Grid Search###################
    grid_param = {'max_depth':[3, 6, 9, 12, 15, 18], 'criterion':['gini','entropy'], 'min_samples_split':list(range(2, 80))}
    
    grid_model =  GridSearchCV(best_model, grid_param, cv = 5, scoring  = 'f1', n_jobs = -1)
    
    grid_model.fit(train_new, y_train_new)
    
    print("\nGrid Search Results:")
    print("best score: %f " % grid_model.best_score_)
    print("best parameters: ")
    print(grid_model.best_params_)
    ####################################################
    
    ########For Kaggle Submission######################
    #Predicting test data
    test = pd.read_csv("../../Data/x_test.csv", header = None)
    test[[0]] = pd.DataFrame(prep.normalize(test[[0]]))
    
    y_test_pred = pd.DataFrame(grid_model.predict(test))
  
    #print(y_test_pred.head)
    
    kaggleize(y_test_pred,"../Predictions/best.csv" )
    print("\nQuestion 3 End")