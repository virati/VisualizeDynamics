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
from sklearn import preprocessing as pproc

from mpl_toolkits.mplot3d import Axes3D

def simple_rad(state):
    r = state[0,:]
    theta = state[1,:]
    
    dot_r = 0*r
    dot_theta = 0.2 + 0*theta
    
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
        
        #This is the main field I think
        self.field = plt.axes([0.05, 0.50, 0.45, 0.45], facecolor='white',projection='polar')
        
        axcolor = 'lightgoldenrodyellow'
        self.axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        self.axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        self.axw = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
        
        self.sfreq = Slider(self.axfreq, 'CFreq', 0, 15.0, valinit=1)
        self.samp = Slider(self.axamp, 'Mu', -10, 8, valinit=1)
        self.sw = Slider(self.axw,'W factor',-0,1.0,valinit=1)
        
        self.sfreq.on_changed(self.update)
        self.samp.on_changed(self.update)
        self.sw.on_changed(self.update)
        
    
    def mouse_coord(self,event):
        mu = self.samp.val
        cfreq = self.sfreq.val
            
        if event.button == 1:
            if event.inaxes == ax:
                global cx,cy,start_loc
                cx,cy = event.xdata, event.ydata
                
                
                t,traj = plot_traj([cx,cy],mu=mu, fc=cfreq)
                scat.remove()
                z = np.linspace(0,traj.shape[0],traj.shape[0])
                scat = ax.scatter(traj[:,0],traj[:,1],color=traj_cmap,alpha=0.8)
                
                curax = plt.axes(tser)
                curax.cla()
                plt.plot(t,traj)
                
                start_loc.remove()
                start_loc = ax.scatter(cx,cy,color='r',marker='>',s=300)
                plt.draw()
            
                self.fig.canvas.draw_idle()
        elif event.button == 3:
            if event.inaxes == ax:
                print('Adding Trajectory Point')
                global trajectory
                trajectory.append([event.xdata,event.ydata])
                traj = np.array(trajectory)
                for ll in range(traj.shape[0]-1):
                    ax.plot(traj[ll:ll+2,0],traj[ll:ll+2,1])
                    
                scat = ax.scatter(traj[:,0],traj[:,1],s=200)
                plt.draw()
                fig.canvas.draw_idle()
                print('trajectory plot')
                
                #Now actually plot the experienced dynamics for the trajectory above
                Ztraj = []
                Zdiff = []
                Thetadiff = []
                
                traj_vect = np.array((traj[-1,0] - traj[0,0],traj[-1,1] - traj[0,1]))
                #traj_vect = traj_vect / np.linalg.norm(traj_vect)
                reptraj_vect = np.tile(traj_vect.reshape(-1,1),40)
                
                for ll in range(traj.shape[0]-1):
                    x_range = np.linspace(traj[ll,0],traj[ll+1,0],40)
                    y_range = np.linspace(traj[ll,1],traj[ll+1,1],40)
                    
                    #Xtr,Ytr = np.meshgrid(x_range,y_range)
                    #XXtr = np.array([Xtr.ravel(),Ytr.ravel()])
                    XYtr = np.vstack((x_range,y_range))
                    
                    Ztr = norm_form(XYtr,[],mu=mu,fc=cfreq,win=w)
                    Ztraj.append(Ztr)
                    
                    Zdiff.append(reptraj_vect - Ztr)
                    Thetadiff.append(np.dot(traj_vect,Ztr))
                    
                curax = plt.axes(tser)
                curax.cla()
                Ztraj = np.array(Ztraj).swapaxes(1,2).reshape(-1,2,order='C')
                #this plots the dynamics field along the trajectory
                plt.plot(Ztraj)
                #this should plot the difference in the trajectory from the intrinsic dynamics at each point
                Zdiff = np.array(Zdiff).swapaxes(1,2).reshape(-1,2,order='C')
                
                Thetadiff = np.array(Thetadiff).reshape(-1,1)
                print(Thetadiff.shape)
                plt.plot(Thetadiff,linestyle='--')
        
    def set_dyn(self,sys_dyn):
        self.dyn_field = sys_dyn
        
    def plot_field(self):
        plt.sca(self.field)
        
        plt.cla()
        mesh_lim = 3
        mesh_res = 20
        rd = np.linspace(-mesh_lim,mesh_lim,mesh_res)
        thd = np.linspace(-mesh_lim,mesh_lim,mesh_res)
        R,TH = np.meshgrid(rd,thd)
        RR = np.array([R.ravel(),TH.ravel()])
        
        Z = np.array(self.dyn_field(RR))
        #pdb.set_trace()
        Z_n = pproc.normalize(Z.T,norm='l2').T
        
        
        self.quiv =  self.field.quiver(TH[:],R[:],np.cos(Z_n[1,:]) - np.sin(Z_n[1,:]),np.sin(Z_n[0,:]) + np.cos(Z_n[0,:]),width=0.01,alpha=0.4)
    
    def mg_compute(self):
        mesh_lim = 3
        mesh_res = 50
        xd = np.linspace(-mesh_lim,mesh_lim,mesh_res)
        yd = np.linspace(-mesh_lim,mesh_lim,mesh_res)
        X,Y = np.meshgrid(xd,yd)
        XX = np.array([X.ravel(),Y.ravel()])
        mu = a0
        cfreq = f0
        w = w0
        Z = np.array(norm_form(XX,[],mu=mu,fc=cfreq,win=w))
        Z_n = pproc.normalize(Z.T,norm='l2').T
    

    def update(self,val):
        #caxis.plot(t,traj)
        #curax = plt.axes(self.field)
        self.plot_field()
        
        
#%%
# main runscript
main_iface = iface()
dynamics = dyn_sys('Hopf')
main_iface.set_dyn(simple_rad)