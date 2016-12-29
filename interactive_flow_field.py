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
import matplotlib.cm as cm
import scipy.signal as sig

from mpl_toolkits.mplot3d import Axes3D

global ax
#fig, ax = plt.subplots()
fig = plt.figure()
tser = plt.axes([0.05, 0.25, 0.90, 0.20], axisbg='white')
phslice = plt.axes([0.5, 0.50, 0.45, 0.45], axisbg='white')

ax = plt.axes([0.05, 0.50, 0.45, 0.45], axisbg='white')
#plt.subplots_adjust(left=0.25, bottom=0.25)



t = np.arange(0.0, 1.0, 0.001)
a0 = 0
f0 = 3
w0 = 0.5
#s = a0*np.sin(2*np.pi*f0*t)
#fieldax = plt.subplot(2,2,1)

global systype
systype = 'SN'

#These are the starting state points
global cx, cy
cx = 4
cy = -2


def crit_points(x,y):
    bcrit_idxs = sig.argrelextrema(np.abs(y),np.less_equal)[0]
    
    #get derivative ready...
    ydiff = np.diff(y)
    stability = np.zeros(shape=bcrit_idxs.shape)
    
    for iiter,ii in enumerate(bcrit_idxs):
        stability[iiter] = np.sign(ydiff[ii])
        
    return bcrit_idxs, stability
    
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

def SN_norm_form(state,t,mu,fc,win):
    x = state[0]
    y = state[1]
    
    #these two can shape peakiness, be used for PAC?

    xd = mu - x**2
    yd = -y
    
    outv = fc * np.array([xd,yd])
    return outv

    
def VDP_norm_form(state,t,mu,fc,win):
    x = state[0]
    y = state[1]
    
    #these two can shape peakiness, be used for PAC?

    xd = win * mu * x - y**2 * x - y
    yd = x
    
    outv = fc * np.array([xd,yd])
    return outv

def norm_form(state,t,mu,fc,win):
    global systype
    if systype == 'Hopf':
        dofunc = H_norm_form
    elif systype == 'VDP':
        dofunc = VDP_norm_form
    elif systype == 'SN':
        dofunc = SN_norm_form
        
    return dofunc(state,t,mu,fc,win)
    
def plot_traj(state0,mu=1,fc=1,win=0.5):
    #t = np.arange(0.0,30.0,0.01)
    t = np.linspace(0.0,10.0,500)
    global cx, cy
    state0 = [cx, cy]
    
    traj = odeint(norm_form,state0,t,args=(mu,fc,win))
    
    return t,traj

mesh_lim = 5
xd = np.linspace(-mesh_lim,mesh_lim,50)
yd = np.linspace(-mesh_lim,mesh_lim,50)
X,Y = np.meshgrid(xd,yd)
XX = np.array([X.ravel(),Y.ravel()])
mu = a0
cfreq = f0
w = w0
Z = np.array(norm_form(XX,[],mu=mu,fc=cfreq,win=w))
Z_n = pproc.normalize(Z.T,norm='l2').T

                     
t,traj = plot_traj([cx,cy],mu=mu,fc=cfreq,win=w)
#for 1d data
#l, = plt.plot(t, s, lw=2, color='red')

#plt.ion()
#

l = plt.quiver(X[:],Y[:],Z_n[0,:],Z_n[1,:],width=0.01,alpha=0.4)
ax.axhline(y=0,color='r')
global scat, start_loc
#plt.subplot(2,2,1)
#z = np.linspace(0,1,t.shape[0])
z = np.linspace(0.0,30.0,500)
traj_cmap = cm.rainbow(z/30)

scat = ax.scatter(traj[:,0],traj[:,1],color=traj_cmap,alpha=0.8,s=20)
#Try to do a continuous trajectory, with colors
#scat = ax.plot(traj[:,0],traj[:,1],alpha=0.8,color=traj_cmap)
start_loc = ax.scatter(cx,cy,color='r',marker='>',s=300)
#set the axes
plt.axis([-5, 5, -5, 5])


caxis = plt.axes(phslice)
Zmag = np.linalg.norm(Z,axis=0).reshape(X.T.shape)[25,:]
crits,stabs = crit_points(xd,Zmag)

plt.plot(xd,Zmag,color='r')
#plt.plot(xd[crits],Zmag[crits],'o',color='red')
plt.scatter(xd[crits],Zmag[crits],color='red')

#Do timeseries plotting
caxis = plt.axes(tser)
caxis.cla()
caxis.plot(t,traj)



#GUI plots now
axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
axamp = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
axw = plt.axes([0.25, 0.05, 0.65, 0.03], axisbg=axcolor)

sfreq = Slider(axfreq, 'CFreq', 0, 15.0, valinit=f0)
samp = Slider(axamp, 'Mu', -10, 10.0, valinit=a0)
sw = Slider(axw,'W factor',-0,1.0,valinit=w0)

plt.draw()

def update(val):
    global scat, start_loc
    mu = samp.val
    cfreq = sfreq.val
    win = sw.val
    
    #this is where the update happens!
    #l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    global cx, cy
    #ps.remove()
    plt.draw()
    
    Z = np.array(norm_form(XX,[],mu=mu,fc=cfreq,win=win))
    Z_n = pproc.normalize(Z.T,norm='l2').T
                
    t,traj = plot_traj([cx,cy],mu=mu, fc=cfreq, win=win)
    
    l.set_UVC(Z_n[0,:],Z_n[1,:])

    z = np.linspace(0.0,30.0,500)
    traj_cmap = cm.rainbow(z/30)
    #l.set_color(traj_cmap[:,z.index(cfreq)])
    
    scat.remove()
    z = np.linspace(0,traj.shape[0],traj.shape[0])
    scat = ax.scatter(traj[:,0],traj[:,1],color=traj_cmap)
    start_loc.remove()
    start_loc = ax.scatter(cx,cy,color='r',marker='>',s=300)
    #p.scatter(traj[:,0],traj[:,1])
    
    curax = plt.axes(tser)
    curax.cla()
    plt.plot(t,traj)
    
    caxis = plt.axes(phslice)
    caxis.cla()
    #Take the middle slice
    Zmag = np.linalg.norm(Z,axis=0).reshape(X.T.shape)[25,:]
    crits,stabs = crit_points(xd,Zmag)
    caxis.plot(xd,Zmag,color='r')
    caxis.scatter(xd[crits],Zmag[crits],color='red')
    
    plt.title('Start: ' + str(cx) + ',' + str(cy))
    plt.draw()
    
    fig.canvas.draw_idle()

sfreq.on_changed(update)
samp.on_changed(update)
sw.on_changed(update)

resetax = plt.axes([-5,5,-5,5])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def get_coord(event):
    #print(str(event.xdata) + ' ' + str(event.ydata))
    #global ax
    if event.inaxes == ax:
        global cx,cy,start_loc,scat
        cx,cy = event.xdata, event.ydata
        mu = samp.val
        cfreq = sfreq.val
        
        t,traj = plot_traj([cx,cy],mu=mu, fc=cfreq)
        scat.remove()
        z = np.linspace(0,traj.shape[0],traj.shape[0])
        scat = ax.scatter(traj[:,0],traj[:,1],color=traj_cmap,alpha=0.8)
        
        curax = plt.axes(tser)
        curax.cla()
        plt.plot(t,traj)
        
        start_loc.remove()
        start_loc = ax.scatter(cx,cy,color='r',marker='>',s=300)
        
cid = fig.canvas.mpl_connect('button_press_event',get_coord)
#cid = fig.canvas.mpl_connect('pick_event',get_coord)

def reset(event):
    sfreq.reset()
    samp.reset()
button.on_clicked(reset)

#rax = plt.axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
#radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


#def colorfunc(label):
#    l.set_color(label)
#    fig.canvas.draw_idle()
#radio.on_clicked(colorfunc)

plt.show()