#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 14:00:03 2016

@author: virati
#inspired/based on code from: http://matplotlib.org/examples/widgets/slider_demo.html
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from sklearn import preprocessing as pproc
from scipy.integrate import odeint
import pdb

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
t = np.arange(0.0, 1.0, 0.001)
a0 = 0
f0 = 3
s = a0*np.sin(2*np.pi*f0*t)

def norm_form(state,t,mu,fc):
    x = state[0]
    y = state[1]
    
    #these two can shape peakiness, be used for PAC?
    w = 0.5
    q = 1-w

    xd = w * (mu * x - y - x * (x**2 + y**2))
    yd = q * (x + mu * y - y * (x**2 + y**2))
    
    outv = fc * np.array([xd,yd])
    return outv
    
def plot_traj(state0,mu=1,fc=1):
    t = np.arange(0.0,30.0,0.01)
    
    traj = odeint(norm_form,state0,t,args=(mu,fc))
    
    return t,traj

mesh_lim = 5
xd = np.linspace(-mesh_lim,mesh_lim,50)
yd = np.linspace(-mesh_lim,mesh_lim,50)
X,Y = np.meshgrid(xd,yd)
XX = np.array([X.ravel(),Y.ravel()])
mu = -1
Z = np.array(norm_form(XX,[],mu=mu,fc=10))
Z_n = pproc.normalize(Z.T,norm='l2').T
            
                     
t,traj = plot_traj([-5,-5])
#for 1d data
#l, = plt.plot(t, s, lw=2, color='red')

plt.ion()
#

l = plt.quiver(X,Y,Z_n[0,:],Z_n[1,:])

global scat
scat = ax.scatter(traj[:,0],traj[:,1])


plt.axis([-5, 5, -5, 5])

axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
axamp = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

sfreq = Slider(axfreq, 'CFreq', 0, 30.0, valinit=f0)
samp = Slider(axamp, 'Mu', -10, 10.0, valinit=a0)

plt.draw()

def update(val):
    global scat
    mu = samp.val
    cfreq = sfreq.val
    #this is where the update happens!
    #l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    
    #ps.remove()
    plt.draw()
    
    Z = np.array(norm_form(XX,[],mu=mu,fc=cfreq))
    Z_n = pproc.normalize(Z.T,norm='l2').T
                
    t,traj = plot_traj([5,5],mu=mu, fc=cfreq)
    
    
    l.set_UVC(Z_n[0,:],Z_n[1,:])
    
    scat.remove()
    scat = ax.scatter(traj[:,0],traj[:,1])
    #p.scatter(traj[:,0],traj[:,1])
    
    plt.draw()
    
    fig.canvas.draw_idle()
    
sfreq.on_changed(update)
samp.on_changed(update)

resetax = plt.axes([-5,5,-5,5])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    sfreq.reset()
    samp.reset()
button.on_clicked(reset)

rax = plt.axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


def colorfunc(label):
    l.set_color(label)
    fig.canvas.draw_idle()
radio.on_clicked(colorfunc)

plt.show()