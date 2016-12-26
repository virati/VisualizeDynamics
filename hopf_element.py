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


from scipy.integrate import odeint

from sklearn import preprocessing as pproc

plt.close('all')


input_val = []

def DEPRHopf(state,t):
    x = state[0]
    y = state[1]
    mu = 1.0
    
    xd = mu * x - y - x * (x**2 + y**2)
    yd = x + mu * y - y * (x**2 + y**2)
    
    return [xd,yd]


class HopfNet():
    params = np.array([])
    flow = []
    def __init__(self):
        self.params = np.array([0,0,0,0,0])
    
    def plot_flow(self):
        xd = np.linspace(-5.0,5.0,100)
        yd = np.linspace(-5.0,5.0,100)
        X,Y = np.meshgrid(xd,yd)
        
        XX = np.array([X.ravel(),Y.ravel()])
        Z = np.array(self.norm_form(XX,t=0,mu=1.0))
        #unit norm the Z vectors
        Z_n = pproc.normalize(Z.T,norm='l2').T
        #Z = Z.reshape(X.T.shape)
                
        plt.figure()
        plt.quiver(X,Y,Z_n[0,:],Z_n[1,:])
        #overlay a trajectory
        traj = self.trajectory([12.0,13.0])
        plt.scatter(traj[:,0],traj[:,1])
        
        plt.show()
        self.flow = Z
    
    def trajectory(self,state0):
        t = np.arange(0.0,30.0,0.01)
        
        traj = odeint(self.norm_form,state0,t)
        
        return traj
        
    def norm_form(self,state,t,mu=1.0):
        x = state[0]
        y = state[1]
        
        xd = mu * x - y - x * (x**2 + y**2)
        yd = x + mu * y - y * (x**2 + y**2)
    
        return [xd,yd]
        
    def DEPRflow_field(self,Z,mu):
        #Z = np.sum(Z,0)
        #Zdot = Z*((lamba + 1j) + b * np.abs(Z)**2)
        
        #hopf normal form? https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwjMw9SN45LRAhUI8mMKHc0SBysQFggcMAA&url=http%3A%2F%2Fmath.arizona.edu%2F~lega%2F454%2FFall06%2FHopf_Bifurcation.pdf&usg=AFQjCNGN5Yo3ENxTo0DPuZLK6oEVXEw5kQ&sig2=eQltwhu0htUC_oGeEnwlGQ
        x = Z[0,:]
        y = Z[1,:]
        
        #xt = mu * Z[0,:] + Z[1,:]
        #yt = -Z[0,:] + mu * Z[1,:] - Z[0,:]**2 * Z[1,:]
        
        xt = mu * x - y - x * (x**2 + y**2)
        yt = x + mu * y - y * (x**2 + y**2)
        
        #Very Simple Identity Linear System
        #xt = Z[0,:]
        #yt = Z[1,:]
        
        #Stack em back and return the vector field
        Z = np.vstack((xt.T,yt.T))
        
        return Z
   
#state0 = [12.0,13.0]
#t = np.arange(0.0,30.0,0.01)
#traj = odeint(Hopf,state0,t)

    

simpleNet = HopfNet()
simpleNet.plot_flow()
traj = simpleNet.trajectory([12.0,13.0])

plt.figure()
plt.scatter(traj[:,0],traj[:,1],c=t)
plt.show()
