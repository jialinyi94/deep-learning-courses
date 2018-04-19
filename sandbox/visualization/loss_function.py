#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:31:36 2018

@author: jialinyi
"""

import numpy as np

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


## Plotting

# Contour

fig, ax = plt.subplots(figsize=(10, 6))

ax.contour(w1_grid, w2_grid, ce_cost_grid, levels=np.logspace(0, 3, 50),
           norm=LogNorm(), cmap=plt.cm.jet)

ax.set_xlabel('$w_1$')
ax.set_ylabel('$w_2$')

ax.set_xlim((w1min, w1max))
ax.set_ylim((w2min, w2max))

plt.show()


# 3D surface plot

fig = plt.figure(figsize=(8, 5))
ax = plt.axes(projection='3d', elev=50, azim=-50)

ax.plot_surface(w1_grid, w2_grid, ce_cost_grid, norm=LogNorm(), 
                rstride=1, cstride=1, 
                edgecolor='none', alpha=.8, cmap=plt.cm.jet)

ax.set_xlabel('$w_1$')
ax.set_ylabel('$w_2$')
ax.set_zlabel('cost')

ax.set_xlim((w1min, w1max))
ax.set_ylim((w2min, w2max))

plt.show()