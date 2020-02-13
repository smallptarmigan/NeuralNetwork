# -*- coding: utf-8 -*-

import sys
import time
import numpy as np
import multiprocessing as mp

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

###################################################

def main():
    X  = np.array([1.0, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])
    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])
    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])

    A1 = np.dot(X, W1) + B1
    Z1 = relu_f(A1)
    A2 = np.dot(Z1, W2) + B2
    Z2 = relu_f(A2)
    A3 = np.dot(Z2, W3) + B3
    Z3 = identify_f(A3)

    y  = softmax(Z3)

    print(y)


if __name__ == "__main__":
    start = time.time()
    main()
    print(">> runtime : {}".format(time.time()-start))
