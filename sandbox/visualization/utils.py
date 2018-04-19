#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:58:23 2018

@author: jialinyi
"""

import numpy as np


# Neural Network

def nn_model(w1, w2, inputs, activation="tanh"):
    '''
    2-layer neural network (linear -> tanh/relu -> linear -> sigmoid)
    '''
    a = inputs
    z = w1 * a
    
    if activation == "tanh":
        a = np.tanh(z)
    elif activation == "relu":
        a = z * (z > 0)
    
    z = w2 * a
    logits = 1 / (1 + np.exp(-z))
    
    return logits


# Loss functions
    
def cross_entropy_loss(labels, logits):
    
    loss = - labels * np.log(logits) - (1 - labels) * np.log(1 - logits)
    
    return loss

def square_loss(labels, logits):
    
    loss = (labels - logits)**2
    
    return loss

def model(inputs, labels, w1, w2,
          activation="tanh", loss_func="cross_entropy"):
    
    if activation == "tanh":
        logits = nn_model(w1, w2, inputs)
    elif activation == "relu":
        logits = nn_model(w1, w2, inputs, activation="relu")
    
    if loss_func == "cross_entropy":
        loss = cross_entropy_loss(labels, logits)
    elif loss_func == "square_loss":
        loss = square_loss(labels, logits)
    
    return np.mean(loss)


def cost(inputs, labels, activation="tanh", loss_func="cross_entropy"):
    
    
    func_obj = lambda w1, w2 : model(inputs, labels, w1, w2, 
                                     activation, loss_func)
    # vectorizing for meshgrid
    vec_func_obj = np.vectorize(func_obj)
    
    return vec_func_obj


# Utility functions
    
def make_minimize_cb(path=[]):
    
    def minimize_cb(xk):
        # note that we make a deep copy of xk
        path.append(np.copy(xk))

    return minimize_cb