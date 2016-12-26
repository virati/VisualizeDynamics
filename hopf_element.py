#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 15:45:43 2016

A simple hopf system that encodes a value by "generating" a limit cycle -> r is the value being encoded
@author: virati
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import pdb

input_val = []

class HopfNet():
    params = np.array([])
    def __init__(self):
        self.params = np.array([0,0,0,0,0])
    
    def plot_flow(self):
        xd = np.linspace(-10,10,100)
        yd = np.linspace(-10,10,100)
        X,Y = np.meshgrid(xd,yd)
        
        XX = np.array([X.ravel(),Y.ravel()]).T
        Z = self.norm_form(XX,1,-5 + 1j)
        
        #Z = Z.reshape(X.T.shape)
        
        plt.figure()
        plt.quiver(X,Y,np.real(Z[0,:]),np.imag(Z[1,:]))
        print(Z)
        plt.show()
        
    def norm_form(self,Z,lamba,b):
        Zdot = Z*((lamba + 1j) + b * np.abs(Z)**2)
        
        return Zdot
        

simpleNet = HopfNet()
simpleNet.plot_flow()