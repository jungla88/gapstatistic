#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:24:42 2020

@author: luca
"""

import gapstatistic
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.spatial.distance import pdist,squareform
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class AgglWrapper:
    
    def __init__(self,affinity="sqeuclidean",linkage = "average"):
        
        self.method = AgglomerativeClustering()        
        self.affinity= affinity
        self.linkage = linkage
    
    def __call__(self,X,k,precomputed=False):
        
        self.method.n_clusters=k
        self.method.affinity = "precomputed" if precomputed else self.affinity        
        self.method.linkage = self.linkage
        
        labels = self.method.fit_predict(X)
        
        return labels

centers = [[1, 1], [-1, -1],[1/2, 1/2], [-1/2, -1/2]]
X, y = make_blobs(n_samples=100, centers=centers, cluster_std=0.1,
                            random_state=0)

X = StandardScaler().fit_transform(X)

plt.scatter(X[:,0],X[:,1])
plt.show()


method = AgglWrapper()

minValCols = np.min(X, axis = 0, keepdims = True)
maxValCols = np.max(X, axis = 0, keepdims = True)

# evaluation1 = gapstatistic.GapStatistic(squareform(pdist(X,metric = "sqeuclidean")),
#                                         method,
#                                         precomputed= True,
#                                         nFeatures = 2,
#                                         minVal=minValCols,
#                                         maxVal = maxValCols,
#                                         distanceFunction="sqeuclidean",
#                                         n_refs = 100,
#                                         kMax=20)

evaluation2 = gapstatistic.GapStatistic(X,
                                        method,
                                        distanceFunction="sqeuclidean",
                                        n_refs = 100,
                                        kMax=20)



#gap1 =evaluation1.evaluateGap()
gap2 = evaluation2.evaluateGap()

plt.plot(list(range(1,21)),gap2[0])
#fig1=plt.scatter(X[:,0],X[:,1])
#fig2=plt.scatter(evaluation1.referenceSets[0][:,0],evaluation1.referenceSets[0][:,1])