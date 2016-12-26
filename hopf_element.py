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
    mu = 1.0
    current_state = [3.0,4.0]
    
    def __init__(self):
        self.params = np.array([0,0,0,0,0])
    
    def plot_flow(self,mu=1.0):
        xd = np.linspace(-5.0,5.0,100)
        yd = np.linspace(-5.0,5.0,100)
        X,Y = np.meshgrid(xd,yd)
        
        XX = np.array([X.ravel(),Y.ravel()])
        self.mu = mu
        
        Z = np.array(self.norm_form(XX,t=0))
        #unit norm the Z vectors
        Z_n = pproc.normalize(Z.T,norm='l2').T
        #Z = Z.reshape(X.T.shape)
                
        plt.figure()
        plt.subplot(211)
        plt.quiver(X,Y,Z_n[0,:],Z_n[1,:])
        
        #overlay a trajectory
        state0 = self.current_state
        
        traj = self.trajectory(state0)
        plt.scatter(traj[:,0],traj[:,1])
        
        plt.subplot(212)
        plt.plot(traj)
        #plt.show()
        
        #the trajectory just ran, so let's just set the last state as the current state
        self.current_state = traj[-1,:]
        
        self.flow = Z
    
    def trajectory(self,state0):
        t = np.arange(0.0,30.0,0.01)
        
        traj = odeint(self.norm_form,state0,t)
        
        return traj
        
    def norm_form(self,state,t):
        x = state[0]
        y = state[1]
        
        mu = self.mu
        
        #these two can shape peakiness, be used for PAC?
        w = 0.5
        q = 1-w
        
        
        
        xd = w * (mu * x - y - x * (x**2 + y**2))
        yd = q * (x + mu * y - y * (x**2 + y**2))
    
        freq_fact = 5
        outv = freq_fact * np.array([xd,yd])
        
        return outv
        
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

import time

if 0:
    for mu in np.linspace(-1.0,10.0,5):
        simpleNet = HopfNet()
        simpleNet.plot_flow(mu=mu)
        #traj = simpleNet.trajectory([12.0,13.0])
    
#%%
#Now we're going to stage our dynamical system -> this is MODELING SWITCHING
#from t=0 to 5 seconds, we'll have mu = -1
#then we switch to mu = 1

#This is SHIT right now
tvect = np.linspace(0,10,10)
simpleNet = HopfNet()

for tt in tvect:
    if tt < 5:
        tmu = -1
    elif tt >= 5:
        tmu = 1
        
    simpleNet.plot_flow(mu=tmu)
    
plt.show()