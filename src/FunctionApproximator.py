# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 07:45:23 2019

@author: kirthi
"""
#%% - Imports
import numpy as np
import argparse
import os, sys
import random
from common import *
import sklearn.pipeline
import sklearn.preprocessing

from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

#%% Function approx class

class FunctionApproximator:
    def __init__(self, numActions, numStates):
        self.approximator= []
        print_debug("Initializing function approximator: RBF Stochastic Gradient Descent Model")
        
        # Build the sample observation data
        sample_data = np.arange(numStates).reshape(-1,1)
        
        # Initialize scaler with 0 mean and 1 variance
        self._normalize = sklearn.preprocessing.StandardScaler()

        self._normalize.fit(sample_data)        
        
        self._rbfFeaturizer = sklearn.pipeline.FeatureUnion(
        [("rbf1", RBFSampler(gamma = 5.0, n_components=100)),
         ("rbf2", RBFSampler(gamma = 2.0, n_components=100)),
         ("rbf3", RBFSampler(gamma = 1.0, n_components=100)),
         ("rbf4", RBFSampler(gamma = 0.5, n_components=100))
        ])
        
        self._rbfFeaturizer.fit(self._normalize.transform(sample_data))
        
        for ind in range(numActions):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([self.rbfFeaturizeState([0])],[0])
            self.approximator.append(model)
            
#    def initializeScaler(self, numStates):
#        minStateValue = 0
#        maxStateValue = numStates - 1
            
    def rbfFeaturizeState(self, state):
        print_debug("Determining RBF feature of Data")
        scaledVal = self._normalize.transform(np.array([state], dtype = np.float64))
        print_debug("Scaled Value for state ",state," is ",scaledVal)
        featurizedVal = self._rbfFeaturizer.transform(scaledVal)
#        print_debug("Feature Value for state ",state," is ",featurizedVal)
        return featurizedVal[0]
        
    def predict(self, state, action=None):
        print_debug("Predicting the output of the state")
        featurizedVal = self.rbfFeaturizeState(state)
        if not action:
            return np.array([ind.predict([featurizedVal])[0] for ind in self.approximator])
        else:
            return self.approximator[action].predict([featurizedVal])[0]
    
    def update(self, state, action, tdTarget):
        print_debug("Updating the output of state")
        featurizedVal = self.rbfFeaturizeState(state)
        self.approximator[action].partial_fit([featurizedVal], [tdTarget])
            