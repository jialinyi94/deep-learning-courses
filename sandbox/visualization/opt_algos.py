#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:25:38 2018

This is a program to demonstrate the effects of common optimization algorithms
on a 2-layer neural network (linear -> tanh/relu -> linear -> sigmoid)

@author: jialinyi
"""

import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm

from utils import *



np.random.seed(1)


## Objective function: cross entropy

samples = 100
prob = 0.3

inputs = np.random.randn(1, samples)
labels = 1 * (np.random.rand(1, samples) > prob)
cross_entropy_cost = cost(inputs, labels)

# meshgrid

w1min, w1max, w1step = -10, 10, .2
w2min, w2max, w2step = -8, 8, .2

w1_list = np.arange(w1min, w1max + w1step, w1step)
w2_list = np.arange(w2min, w2max + w2step, w2step)
w1_grid, w2_grid = np.meshgrid(w1_list, w2_list)

ce_cost_grid = cross_entropy_cost(w1_grid, w2_grid)


## Optimization

# initialization

weights0 = [-7.5, -6]

# numerical optimization

cross_entropy_cost_list = lambda weights: cross_entropy_cost(*weights)


path_ = [weights0]
res = minimize(cross_entropy_cost_list, x0=weights0,
               tol=1e-20, callback=make_minimize_cb(path_))
path = np.array(path_).T


## Plotting


# Contour

fig, ax = plt.subplots(figsize=(10, 6))

ax.contour(w1_grid, w2_grid, ce_cost_grid, levels=np.logspace(0, 3, 50),
           norm=LogNorm(), cmap=plt.cm.jet)

ax.quiver(path[0,:-1], path[1,:-1], 
          path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], 
          scale_units='xy', angles='xy', scale=1, color='k')

ax.set_xlabel('$w_1$')
ax.set_ylabel('$w_2$')

ax.set_xlim((w1min, w1max))
ax.set_ylim((w2min, w2max))

plt.show()