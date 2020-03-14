# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 00:31:20 2019

@author: nitin
"""

import numpy as np
import matplotlib.pyplot as plt


X_train = np.load('../../Data/X_train.npy')
y_train = np.load('../../Data/y_train.npy')

mu = np.mean(X_train, axis = 0)

#std = np.std(X_train, axis = 0)

Z = (X_train - mu) #/ std

n = Z.shape[0]

cov_x = np.cov(Z, rowvar = False)

w, v = np.linalg.eig(cov_x)

idx = np.argsort(-w)

eig = w[idx]
P = v[:,idx]

# Question 2.1
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(np.arange(len(eig)),eig)
plt.ylabel("eigenvalues")
plt.xlabel("number of features")
plt.title("(Q2.1) eigenvalues vs number of features")
plt.savefig("../Figures/q2.1_eigenvalue.png")
##################################################

# Question 2.2
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mu.reshape(28,28))
ax.axis("off")
plt.title("(Q2.2) Mean Image of Train Dataset")
plt.savefig("../Figures/q2.2_meandataset.png")



fig = plt.figure(figsize = (12,12))
for i in range(10):
    X_i = X_train[y_train==i]
    mu_i = np.mean(X_i, axis = 0)
    
    ax = fig.add_subplot(4, 3, i+1)
    ax.imshow(mu_i.reshape(28,28))
    ax.axis("off")
    title = "Mean Image of digit " + str(i) + " of train dataset"
    plt.title(title, fontsize = 11)
plt.savefig("../Figures/q2.2_meandigit.png")
##################################################

# Question 2.3
fig = plt.figure(figsize = (10,10))
for i in range(5):
    ax = fig.add_subplot(2, 3, i+1)
    e_i = v[:,i].real
    ax.imshow(e_i.reshape(28,28))
    ax.axis("off")
    title = "eigenvector " + str(i+1)
    plt.title(title)
plt.savefig("../Figures/q2.3_topeigenvector.png")  
##################################################

# Question 2.4
P_i = P[:,:2]
Z_k = Z.dot(P_i)
colordict = {0:'red', 1:'orange', 2:'yellow', 3:'green', 4:'turquoise', 5:'cyan', 6:'blue', 7:'purple', 8:'magenta', 9:'brown'}

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for i in np.unique(y_train):
    idx = np.where(y_train == i)
    ax.scatter(Z_k[idx,0], Z_k[idx,1], c=colordict[i], label = i)
plt.xlabel("component 1")
plt.ylabel("component 2")
plt.title("Plot of MNIST train dataset projected to 2 components")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig("../Figures/q2.4_projComponents.png", bbox_inches='tight')  
##################################################

