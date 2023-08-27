#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 21:55:06 2017
This script will attempt to do DMD for decomposition of the 2d vector field
based on/learned from: http://www.pyrunner.com/weblog/2016/07/25/dmd-python/
@author: virati
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import dot, multiply, diag, power
from numpy import pi, exp, sin, cos, cosh, tanh, real, imag
from numpy.linalg import inv, eig, pinv
from scipy.linalg import svd, svdvals
from scipy.integrate import odeint, ode, complex_ode
from warnings import warn


def nullspace(A, atol=1e-13, rtol=0):
    # from http://scipy-cookbook.readthedocs.io/items/RankNullspace.html
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

def check_linear_consistency(X, Y, show_warning=True):
    # tests linear consistency of two matrices (i.e., whenever Xc=0, then Yc=0)
    A = dot(Y, nullspace(X))
    total = A.shape[1]
    z = np.zeros([total, 1])
    fails = 0
    for i in range(total):
        if not np.allclose(z, A[:,i]):
            fails += 1
    if fails > 0 and show_warning:
        warn('linear consistency check failed {} out of {}'.format(fails, total))
    return fails, total

def dmd(X, Y, truncate=None):
    U2,Sig2,Vh2 = svd(X, False) # SVD of input matrix
    r = len(Sig2) if truncate is None else truncate # rank truncation
    U = U2[:,:r]
    Sig = diag(Sig2)[:r,:r]
    V = Vh2.conj().T[:,:r]
    Atil = dot(dot(dot(U.conj().T, Y), V), inv(Sig)) # build A tilde
    mu,W = eig(Atil)
    Phi = dot(dot(dot(Y, V), inv(Sig)), W) # build DMD modes
    return mu, Phi

def check_dmd_result(X, Y, mu, Phi, show_warning=True):
    b = np.allclose(Y, dot(dot(dot(Phi, diag(mu)), pinv(Phi)), X))
    if not b and show_warning:
        warn('dmd result does not satisfy Y=AX')
        
# convenience method for integrating
def integrate(rhs, tspan, y0):

    # empty array to hold solution
    y0 = np.asarray(y0) # force to numpy array
    Y = np.empty((len(y0), tspan.size), dtype=y0.dtype)
    Y[:, 0] = y0

    # auto-detect complex systems
    _ode = complex_ode if np.iscomplexobj(y0) else ode

    # create explicit Runge-Kutta integrator of order (4)5
    r = _ode(rhs).set_integrator('dopri5')
    r.set_initial_value(Y[:, 0], tspan[0])

    # run the integration
    for i, t in enumerate(tspan):
        if not r.successful():
            break
        if i == 0:
            continue # skip the initial position
        r.integrate(t)
        Y[:, i] = r.y

    # return solution
    return Y

# the right-hand side of our ODE
def saddle_focus(t, y, rho = 1, omega = 1, gamma = 1):

    return dot(np.array([
                [-rho, -omega, 0],
                [omega, -rho, 0],
                [0, 0, gamma]
            ]), y)

# generate data
dt = 0.01
tspan = np.arange(0, 50, dt)
X = np.zeros([3, 0])
Y = np.zeros([3, 0])
z_cutoff = 15 # to truncate the trajectory if it gets too large in z
for i in range(20):
    theta0 = 2*pi*np.random.rand()
    x0 = 3*cos(theta0)
    y0 = 3*sin(theta0)
    z0 = np.random.randn()
    D = integrate(saddle_focus, tspan, [x0, y0, z0])
    D = D[:,np.abs(D[2,:]) < z_cutoff]
    X = np.concatenate([X, D[:,:-1]], axis=1)
    Y = np.concatenate([Y, D[:,1:]], axis=1)

check_linear_consistency(X, Y); # raise a warning if not linearly consistent

mu, Phi = dmd(X,Y,3)
check_dmd_result(X,Y,mu,Phi)
#%%

plt.figure()
plt.plot(X.T)
plt.show()

