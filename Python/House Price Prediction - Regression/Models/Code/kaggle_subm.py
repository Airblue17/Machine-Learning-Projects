# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:14:51 2019

@author: nitin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import *
from sklearn.model_selection  import *
from sklearn.ensemble import *
from sklearn.tree import DecisionTreeRegressor as dtree  
from sklearn import preprocessing as prep
from write_csv import *
from sklearn.decomposition import PCA
import random
from sklearn.linear_model import *
import xgboost as xgb

def Kaggle_Submission():
    random.seed(99)
    x_train = pd.read_csv("../../Data/train_x.csv", header = None)
    y_train = pd.read_csv("../../Data/train_y.csv", header = None)
    
    
    non_outlier_idx = y_train.index[y_train[0] <= 600000].tolist()
    
    #creating a new temporary non outlier data frame
    x_train = x_train.iloc[non_outlier_idx,:]
    y_train = y_train.iloc[non_outlier_idx,:]
    x_train = x_train.assign(labels = y_train).sample(frac = 1, random_state = 17)
    y_train = x_train['labels']
    del x_train['labels']
    
    #####For SVM
    C = [0.0001, 0.001, 0.01, 0.3, 1]
    
    gamma = [0.001, 0.03, 0.1, 0.3, 1, 3]
    
    kernel = ['linear','rbf']
    
    grid_param_svm = {'C':C, 'gamma':gamma, 'kernel':kernel}
    
    svm_model = GridSearchCV(estimator=SVR(), param_grid=grid_param_svm, cv = 5, n_jobs=-1, scoring = 'neg_mean_absolute_error')
    
    # Train the classifier 
    svm_model.fit(x_train, y_train.values.ravel())   
    
    ######
    
    ####Random Forest
    estimators = np.arange(10, 100, 10).tolist()
    grid_param_rf = {'n_estimators':estimators}
    
    rf_model = GridSearchCV(estimator=RandomForestRegressor(n_jobs=-1, random_state = 1), param_grid=grid_param_rf, cv = 5, n_jobs=-1, scoring = 'neg_mean_absolute_error')
    
    rf_model.fit(x_train, y_train.values.ravel())  
    ###
    
    
    
    ###XGBoost
    gamma = [0.001, 0.03, 0.1, 0.3, 1, 3]
    learning_rate =  [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.3]
    grid_param_xgb = {'n_estimators':estimators, 'gamma':gamma, 'learning_rate':learning_rate}
    
    xgb_model = GridSearchCV(estimator = xgb.XGBRegressor(objective="reg:linear", random_state=42), param_grid = grid_param_xgb, cv = 5, n_jobs=-1, scoring = 'neg_mean_absolute_error')
    ###
    
    
    ###DTree
    grid_param_dtree = {'max_depth':[3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33], 'max_features':['auto','sqrt', 'log2'], 'min_samples_split':list(range(2, 80))}
        
    dtree_model =  GridSearchCV(dtree(random_state = 17), grid_param_dtree, cv = 5, scoring  = 'neg_mean_absolute_error', n_jobs = -1)
        
    dtree_model.fit(x_train, y_train.values.ravel())
    ###
    
    ###Lasso
    alpha = [0.003, 0.01, 0.03, 0.1, 0.3, 1]
    
    grid_param_lasso= {'alpha':alpha}
    lasso_model = GridSearchCV(estimator=Lasso(max_iter=10e5), param_grid=grid_param_lasso, cv = 5, n_jobs=-1, scoring = 'neg_mean_absolute_error')
    
    lasso_model.fit(x_train, y_train.values.ravel())
    ###
    
    reg_model1 = svm_model
    reg_model2 = rf_model
    reg_model3 = dtree_model
    reg_model4 = lasso_model
    reg_model5 = xgb_model
    ensemble_model = VotingRegressor(estimators=[('svr', reg_model1), ('rf', reg_model2), ('dtree', reg_model3), ('lasso', reg_model4), ('xgb', reg_model5)])
    
    ensemble_model.fit(x_train, y_train.values.ravel())
    
    #Predicting test data
    test = pd.read_csv("../../Data/test_x.csv", header = None)
    
    test.fillna(test.mean() ,inplace=True) 
    
    y_test_pred = pd.DataFrame(ensemble_model.predict(test))
    
    y_test_pred = y_test_pred.to_numpy().reshape(test.shape[0],)
      
    #print(y_test_pred.head)
    
    write_csv(y_test_pred,"../Predictions/best.csv" )
        