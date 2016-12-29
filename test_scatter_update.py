#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 14:33:05 2016

@author: virati
"""

import matplotlib.pyplot as plt
import numpy as np

plt.ion()

x = np.random.normal(size=(500,1))
y = np.random.normal(loc=2,size=(500,1))

plt.figure()
ps = plt.scatter(x,y)
plt.draw()

plt.pause(1)

x = np.random.normal(loc=10,size=(500,1))
y = np.random.normal(loc=-2,size=(500,1))
ps.remove()
ps.set_array(np.hstack((x,y)))
plt.draw()

#%%


#plt.clf()
plt.scatter(x,y)
plt.draw()

plt.show()