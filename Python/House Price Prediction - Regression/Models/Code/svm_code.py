# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 23:24:10 2019

@author: nitin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn import metrics, preprocessing as prep
import random
from kaggle import *
from sklearn.model_selection  import *

def SVM_code():
    
    #loading the data
    x_train = pd.read_csv("../../Data/train_x.csv", header = None)
    y_train = pd.read_csv("../../Data/train_y.csv", header = None)
    
    random.seed(567)
    
    #Plotting the house price prediction as histograms
    plt.subplots(1)
    plt.hist(y_train.to_numpy(), bins = 50)
    plt.show()
    
    #print(np.sum(y_train_full>600000))
    
    x_train_new = x_train
    y_train_new = y_train
    
    #Grid Search Parameters
    parameter_candidates = [
      {'C': [1, 0.01, 0.0001], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf']},
      {'C': [1, 0.01, 0.0001], 'gamma': [0.1], 'kernel': ['linear']},
    ]
    
    model = GridSearchCV(estimator=SVR(), param_grid=parameter_candidates, cv = 5, n_jobs=-1, scoring = 'neg_mean_absolute_error')
    
    # Train the classifier 
    model.fit(x_train_new, y_train_new.values.ravel())   
    
    print()
    for i in range(model.cv_results_['mean_test_score'].size):
        print("Parameters:")
        print(model.cv_results_['params'][i])
        print("MAE:")
        print(model.cv_results_['mean_test_score'][i])
        print()
    
    
