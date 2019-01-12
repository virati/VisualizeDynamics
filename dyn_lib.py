#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 21:45:54 2019

@author: virati
Library of dynamics functions
"""
import numpy as np

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


def mod_H(state,t,mu,fc,win=0.5):
    x = state[0]
    y = state[1]
    
    z = x + 1j * y
    
    zdot = -z * (-mu + 2*np.abs(z)**2 + w*1j + 1j * np.abs(z)**2)
    xchange = (np.real(zdot))
    ychange = (np.imag(zdot))
    
    fc=1
    outv = fc * np.array([xchange,ychange])
    return outv

def var_H(state,t,mu,fc,win=0.5):
    x = state[0]
    y = state[1]
    
    z = x + 1j * y
    w = 0.2
    #zdot = -z * (-mu + 2*np.abs(z)**2 + 1j * z * (fc*w/np.abs(z)**2))
    zdot = -z * (-mu + 2*np.abs(z)**2 + w*1j - 10*1j * np.abs(z)**2)
    xchange = (np.real(zdot))
    ychange = (np.imag(zdot))
    
    fc=1
    outv = fc * np.array([xchange,ychange])
    return outv


def SN_norm_form(state,t,mu,fc,win):
    x = state[0]
    y = state[1]
    
    #these two can shape peakiness, be used for PAC?

    xd = mu - x**2
    yd = -y
    
    outv = fc * np.array([xd,yd])
    return outv

def global_norm_form(state,t,mu,fc,win):
    x = state[0]
    y = state[1]
    
    #these two can shape peakiness, be used for PAC?

    xd = y
    yd = mu * y + x - x**2 + x * y
    
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