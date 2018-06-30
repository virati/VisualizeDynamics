#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 14:29:14 2018

@author: virati
This file will be focused on $x$ and $u$ for a given dynamics. We'll do things like find a u that gives us a desired x trajectory, finding the x traj for a given u, etc.
All backend stuff, should plug into interactive flow field eventually
"""

import sympy
import numpy
#import networkx
import matplotlib.pyplot as plt

def H_norm_form(state,t,mu,fc,win=0.5):
    x = state[0]
    y = state[1]
    
    #these two can shape peakiness, be used for PAC?
    w = win
    q = 1-w

    xd = w * (mu * x - y - x * (x**2 + y**2))
    yd = q * (x + mu * y - y * (x**2 + y**2))
    
    outv = fc * np.array([xd,yd])
    return outv

class OptControl:
    
    def __init__(self):
        self.active = True
        
    def dyn_eq(self,state,time,mu,fc,win):
        return 0
        
    def run_sim(self):
        return 0
        
    def get_traj(self,u):
        #t = np.arange(0.0,30.0,0.01)
        t = np.linspace(0.0,10.0,500)
        global cx, cy
        state0 = [cx, cy]
        
        traj = odeint(self.dyn_eq,state0,t,args=(mu,fc,win))
        
        return t,traj
        
    def get_u(self,x_traj):
        return 0
    
    