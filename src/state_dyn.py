#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 13:09:17 2018

@author: virati
Example statespace to measurement space
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as sig

class state_space:
    t = []
    x = []
    def __init__(self,dim=2):
        tsteps = 1000
        t = np.linspace(0,1,tsteps)
        x = np.zeros((tsteps,2))
    
    def map_model_1(state_in):
        return state_out

    def map_model_2(state_in):
        y = np.sin(2 * np.pi * state_in[:,0] * t) + np.sin(2 * np.pi * state_in[:,1] * t)
        return y
    
    def mapper(self):
        
    def measure(self):
        
        
if __name__=='__main__':
    Sp = state_space()