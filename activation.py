# -*- coding: utf-8 -*-

import numpy as np

def step_f(x):
    y = x > 0
    return y.astype(np.float)

def sigmoid_f(x):
    return 1 / (1 + np.exp(-x))

def relu_f(x):
    return np.maximum(0, x)

def identify_f(x):
    return x

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(c - a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

    