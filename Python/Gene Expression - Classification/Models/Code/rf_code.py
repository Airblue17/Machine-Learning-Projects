# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 00:30:27 2019

@author: nitin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing as prep
import random
from sklearn.model_selection import *
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.tree import *

def ensemble_part():
    random.seed(99)
    print("QUESTION 2 \n")
    x_train = pd.read_csv("../../Data/gene_data/gene_train_x.csv", header = None)
    y_train = pd.read_csv("../../Data/gene_data/gene_train_y.csv", header = None)
    
    x_test = pd.read_csv("../../Data/gene_data/gene_test_x.csv", header = None)
    y_test = pd.read_csv("../../Data/gene_data/gene_test_y.csv", header = None)
    
    num_features = x_train.shape[1]
    
    estimators = np.arange(1, 151, 1).tolist()
    
    features = [num_features//3, num_features, int(np.sqrt(num_features))]
    
    classification_error = {}
    
    fig, ax = plt.subplots(1)
    
    x = estimators
    #########################Graph for Random Forest
    for feature_size in features:
        classification_error[feature_size] = []
        for estimator_size in estimators:
            rf_model = RandomForestClassifier(n_jobs=-1, random_state = 1, n_estimators = estimator_size, max_features = feature_size)
            
            rf_model.fit(x_train, y_train.values.ravel())
            
            y_pred = rf_model.predict(x_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            classification_error[feature_size].append(1 - accuracy)
        
        y = classification_error[feature_size]
        label_name = "Feature Size: "+str(feature_size)
        plt.plot(x, y, label=label_name)
    plt.xticks(np.arange(1, 151, 15))
    plt.xlabel("Number of Trees")
    plt.ylabel("Test Classification Error")
    plt.title("Graph for Test Classification Error vs Number of Trees (Random Forest)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show(fig)
    fig.savefig("../Figures/rfTCEvsNumTrees.png", bbox_inches='tight')
    ################################################################
    
    
    ########Hyperparameter Tuning for Random Forest################
    grid_param_rf = {'n_estimators':estimators, 'max_features': features}
    grid_rf_model = GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1, random_state = 1), param_grid=grid_param_rf, cv = 5, n_jobs=-1, scoring = 'accuracy')
    grid_rf_model.fit(x_train, y_train.values.ravel()) 
    
    #prediction
    y_pred = grid_rf_model.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Test Classification Error using Random Forest: " + str(1-accuracy))
    print()
    ##############################################################
    
    
    #############Graph for Ada Boost##############################
    classification_error = {}
    
    fig, ax = plt.subplots(1)
    
    base_estimators = [DecisionTreeClassifier(max_depth=1),  DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=3)]
    
    m_depth = 1
    for b_estimator in base_estimators:
        classification_error[m_depth] = []
        for estimator_size in estimators:
            adaboost_model = AdaBoostClassifier(base_estimator = b_estimator, random_state = 1, n_estimators = estimator_size, learning_rate = 0.1 )
            
            adaboost_model.fit(x_train, y_train.values.ravel())
            y_pred = adaboost_model.predict(x_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            classification_error[m_depth].append(1 - accuracy)
        
        y = classification_error[m_depth]
        label_name = "Base Estimator Max Depth:  "+str(m_depth)
        m_depth += 1
        plt.plot(x, y, label=label_name)
    plt.xticks(np.arange(1, 151, 15))
    plt.xlabel("Number of Trees")
    plt.ylabel("Test Classification Error")
    plt.title("Graph for Test Classification Error vs Number of Trees (AdaBoost)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show(fig)
    fig.savefig("../Figures/adaboostTCEvsNumTrees.png", bbox_inches='tight')
    ####################################################################
    
    ########Hyperparameter Tuning for Adaboost################
    grid_param_ab = {'n_estimators':estimators, 'base_estimator': base_estimators}
    grid_ab_model = GridSearchCV(estimator=AdaBoostClassifier(random_state = 1, learning_rate = 0.1 ), param_grid=grid_param_ab, cv = 5, n_jobs=-1, scoring = 'accuracy')
    grid_ab_model.fit(x_train, y_train.values.ravel()) 
    
    #prediction
    y_pred = grid_ab_model.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Test Classification Error using AdaBoost: " + str(1-accuracy))
    print()
    ##############################################################
    
    
    ##########Hyperparameter tuning for Decision Tree################
    grid_param_dtree = {'max_depth':[3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33], 'max_features':['auto','sqrt', 'log2'], 'min_samples_split':list(range(2, 100))}
    dtree_model =  GridSearchCV(DecisionTreeClassifier(random_state = 17), grid_param_dtree, cv = 5, scoring  = 'accuracy', n_jobs = -1)
    dtree_model.fit(x_train, y_train.values.ravel())
    
    #prediction
    y_pred = dtree_model.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Test Classification Error using Decision Tree: " + str(1-accuracy))
    print("\n QUESTION 2 END \n")
    ################################################################