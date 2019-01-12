#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 05:00:25 2019

@author: virati
Radial version of interactive flow field
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from sklearn import preprocessing as pproc
from scipy.integrate import odeint
import pdb
import matplotlib.cm as cm
import scipy.signal as sig

from mpl_toolkits.mplot3d import Axes3D

def simple_rad(state):
    r = state[0]
    theta = state[1]
    
    dot_r = 0
    dot_theta = 0.2
    
    return np.array([dot_r,dot_theta])

class dyn_sys:
    def __init__(self,systype):
        pass
        self.set_sys(systype)

    def set_sys(self,systype='Hopf'):
        self.sys_type = systype
        if systype == 'Hopf':
            self.sys_dyn = simple_rad
        
    def compute_traj(self,start):
        pass
    
class iface:
    def __init__(self):
        self.fig = plt.figure()
        self.tser = plt.axes([0.05, 0.25, 0.90, 0.20], facecolor='white')
        self.phslice = plt.axes([0.5, 0.50, 0.45, 0.45], facecolor='white',projection='3d')
        
        self.ax = plt.axes([0.05, 0.50, 0.45, 0.45], facecolor='white')
        
    def set_dyn(self,sys_dyn):
        self.dyn_field = sys_dyn
        
    def plot_field(self):
        pass
        
#%%
# main runscript
main_iface = iface()
dynamics = dyn_sys('Hopf')
main_iface.set_dyn(dynamics)