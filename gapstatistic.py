#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import pdist

class GapStatistic:
    
    def __init__(self,
                 X,
                 clusteringMethod,
                 distanceFunction,
                 nFeatures= None,
                 precomputed = False,
                 minVal=None,
                 maxVal = None,
                 n_refs = 10,
                 kMax = 10):
        
        self._n_refs = n_refs
        self._clusteringMethod = clusteringMethod
        self.distanceFunction = distanceFunction 
        self._kMax = kMax
        self.__precomputed = precomputed
        self._minValCols=minVal
        self._maxValCols=maxVal
        
        #convert to np if not
        if not isinstance(X,np.ndarray):
            X = np.asarray(X)
        
        #check precompute or distance function. 
        if precomputed==True:
            assert(X.shape[0] == X.shape[1])
            assert(nFeatures)
            assert(minVal.all() and maxVal.all())
            features = nFeatures
        else:
            features = X.shape[1]
           
        assert(clusteringMethod)
        
        #data assumed be N pattern by F num_features matrix
        self.data = X
        self.__numObservations = X.shape[0]
        self.__numFeatures = features

        #Setup references distribution
        self._referenceSets = None
        if not precomputed: # and if ref == "uniform"
            self.__setupUniformRefs()
        #else
        #setup PCA
        #...
        
        self._logWdata=None
        self._ExpectedlogW_nrefs=None
        
    @property
    def referenceSets(self):
        return self._referenceSets
    
    @property
    def logWdata(self):
        return self._logWdata
    
    @property
    def E_logWrefs(self):
        return self._ExpectedlogW_nrefs
    
    @property
    def kMax(self):
        return self._kMax
    @kMax.setter
    def kMax(self,val):
        if val > 0:
            self._kMax = val 
            
    @property
    def n_refs(self):
        return self._n_refs
    @n_refs.setter
    def n_refs(self,val):
        if val>0:
            self._n_refs = val
            
    @property
    def minVal(self):
        return self._minValCols
    
    @property
    def maxVal(self):
        return self._maxValCols
    
    def evaluateGap(self):
        
        #Data init
        logW_ref = np.zeros((self._n_refs,self._kMax))
        logW_data = np.zeros((1,self._kMax))       
        self._referenceSets = [np.zeros((self.__numObservations,self.__numFeatures)) for _ in range(self._n_refs)]
        #Reference loops
        for i in range(self._n_refs):
            self._referenceSets[i] = self.__generateReferences()
            
            for k in range(self._kMax):
                labels = self._clusteringMethod(self._referenceSets[i],k+1, precomputed = False)                
                clusters = [self._referenceSets[i][labels==l] for l in range(k+1)]
                clusters_sumofdist = [sum(pdist(cluster,metric = self.distanceFunction)/len(cluster)) 
                                      for cluster in clusters]                
                logW_ref[i][k] = np.log(sum(clusters_sumofdist))                
         
        
        self._ExpectedlogW_nrefs = np.mean(logW_ref, axis = 0, keepdims = True)
        
        #Data loop
        for k in range(self._kMax):
            labels = self._clusteringMethod(self.data, k+1, precomputed = self.__precomputed)
            if self.__precomputed:
                clusters_distMat = [self.data[l==labels][:,l==labels] for l in range(k+1)]
                sumofdist = [0.5*np.sum(cluster,axis=(0,1))/len(cluster) for cluster in clusters_distMat]
            else:
                clusters = [self.data[labels==l] for l in range(k+1)]
                sumofdist = [sum(pdist(cluster,metric = self.distanceFunction)/(len(cluster)))
                             for cluster in clusters]
            
            logW_data[0][k] = np.log(sum(sumofdist))

        self._logWdata = logW_data
        
        gap_k = self._ExpectedlogW_nrefs - self._logWdata
        
        return gap_k
                
    def __generateReferences(self):
        
        return np.random.uniform(low= self._minValCols,high=self._maxValCols,size=(self.__numObservations,self.__numFeatures))

    
    def __setupUniformRefs(self):
        
        self._minValCols = np.min(self.data, axis = 0, keepdims = True)
        self._maxValCols = np.max(self.data, axis = 0, keepdims = True)
        
        self._referenceSets = [np.zeros((self.__numObservations,self.__numFeatures)) for _ in range(self.n_refs)]