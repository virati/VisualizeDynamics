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
import time

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
    current_state = [1.5,0.0]
    fc = 0
    traj = {}
    rad = 0
    
    def __init__(self,center_freq=5,radius=1.5):
        self.params = np.array([0,0,0,0,0])
        self.fc = center_freq
        self.rad = radius
        #mu now needs to be a function of the desired/input radius
        self.mu = radius
    
    def plot_flow(self):
        
        mesh_lim = 5
        xd = np.linspace(-mesh_lim,mesh_lim,50)
        yd = np.linspace(-mesh_lim,mesh_lim,50)
        X,Y = np.meshgrid(xd,yd)
        
        XX = np.array([X.ravel(),Y.ravel()])
        mu = self.mu
        
        Z = np.array(self.norm_form(XX,t=0))
        #unit norm the Z vectors
        Z_n = pproc.normalize(Z.T,norm='l2').T
        #Z = Z.reshape(X.T.shape)
                
        plt.figure()
        plt.subplot(211)
        plt.quiver(X,Y,Z_n[0,:],Z_n[1,:])
        
        #overlay a trajectory
        state0 = self.current_state
        
        tvect,traj = self.trajectory(state0)
        plt.scatter(traj[:,0],traj[:,1])
        plt.xlim((-5,5))
        plt.ylim((-5,5))
        plt.axis('tight')
        
        
        plt.subplot(212)
        plt.plot(tvect,traj)
        #plt.show()
        
        
        
        #the timeseries of the trajectory
        self.traj = {'X':traj,'T':tvect}
        #the trajectory just ran, so let's just set the last state as the current state
        #self.current_state = traj[-1,:]
        
        self.flow = Z
    
    def tf_traj(self):
        #do TF analyses on trajectory
        tvect = self.traj['T']
        X = self.traj['X']
        
        plt.figure()
        plt.subplot(121)
        F,T,SG = sig.spectrogram(X[:,0],nperseg=512,noverlap=256,window=sig.get_window('blackmanharris',512),fs=100)
        plt.pcolormesh(T,F,10*np.log10(SG))  
        plt.subplot(122)
        F,T,SG = sig.spectrogram(X[:,1],nperseg=512,noverlap=256,window=sig.get_window('blackmanharris',512),fs=100)
        plt.pcolormesh(T,F,10*np.log10(SG))  
        
    def trajectory(self,state0):
        t = np.arange(0.0,30.0,0.01)
        
        traj = odeint(self.norm_form,state0,t)
        
        return t,traj
        
    def norm_form(self,state,t):
        x = state[0]
        y = state[1]
        
        mu = self.mu
        
        #these two can shape peakiness, be used for PAC?
        w = 0.5
        q = 1-w
        
        
        
        xd = w * (mu * x - y - x * (x**2 + y**2))
        yd = q * (x + mu * y - y * (x**2 + y**2))
    
        freq_fact = self.fc
        
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

def main():
    
    if 1:
        for mu in [2.0]:
            simpleNet = HopfNet(center_freq=140,radius=10)
            
            simpleNet.plot_flow()
            #traj = simpleNet.trajectory([12.0,13.0])
            simpleNet.tf_traj()
    ##%%
    #How do we actually change the "radius" of our limit cycle?
    #This could/will correspond to the overall binarization/discrimination of the system
    #center freq is DIFFERENT -> gives us an idea of how to go around in time; adjustst the flow field amplitudes, but once the limit cycle reached, it just is
    #R is different -> gives us a 2dimensional system
    #Where one is the limit cycle r; the other is the \dot{theta}
    
    
    
    ##Now we're going to stage our dynamical system -> this is MODELING SWITCHING
    ##from t=0 to 5 seconds, we'll have mu = -1
    ##then we switch to mu = 1
    #
    ##This is SHIT right now
    #tvect = np.linspace(0,10,10)
    #simpleNet = HopfNet()
    #
    #for tt in tvect:
    #    if tt < 5:
    #        tmu = -1
    #    elif tt >= 5:
    #        tmu = 1
    #        
    #    simpleNet.plot_flow(mu=tmu)
    #    
    plt.show()
    
main()