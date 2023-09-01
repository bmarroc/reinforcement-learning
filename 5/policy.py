#!/usr/bin/env python

"""Class that defines the model for the policy.
"""

import numpy as np

class Policy():
    
    def __init__(self, q, epsilon, policy_info={}):
        self.rand_generator = np.random.RandomState(policy_info.get("seed"))
        self.q = q
        self.epsilon = epsilon
                
    def predict(self, inputs):
        s = np.repeat(inputs, repeats=2, axis=0) 
        
        a = np.tile(np.arange(0,2), [inputs.shape[0]]).reshape((-1,1))
        
        y1 = np.concatenate((s,a), axis=1)
        
        y2 = self.q.predict(y1)
        
        y3 = y2.reshape((inputs.shape[0],-1))
        
        y4 = np.max(y3, axis=1).reshape((-1,1))
        
        y5 = 1-np.square(np.sign(y3-y4))
        
        y6 = y5/np.sum(y5, axis=1, keepdims=True)

        y7 = []
        for s in range(y6.shape[0]):
            y7.append([self.rand_generator.choice(2, p=y6[s])])
        y7 = np.array(y7)
        
        p = np.zeros((y7.shape[0],2))
        for s in range(p.shape[0]) :
            p[s][y7[s][0]] = 1.
            
        d1 = self.epsilon/2
        d2 = 1-(2-1)*d1
        
        p = d1*(np.ones(2)-p) + d2*p

        return p