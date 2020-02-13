# -*- coding: utf-8 -*-

import sys
import time
import numpy as np
import multiprocessing as mp

import dataset
import network
import activation as act_f

###################################################

BATCH_SIZE = 100

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
    Z1 = act_f.relu_f(A1)
    A2 = np.dot(Z1, W2) + B2
    Z2 = act_f.relu_f(A2)
    A3 = np.dot(Z2, W3) + B3
    Z3 = act_f.identify_f(A3)

    y  = act_f.softmax(Z3)

    print(y)


if __name__ == "__main__":
    start = time.time()
    main()
    print(">> runtime : {}".format(time.time()-start))
