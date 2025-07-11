#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 09:14:33 2025

@author: danielWH
"""

import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
import math
from scipy import stats
rng = np.random.default_rng()
fig, R = plt.subplots()

a = 1                            # - Hopping Constant
n = 500                          # - #Sites
m = np.arange(0.02, 0.21, 0.01)  # - Disorder Coefficient
k = 100                          # - #Trials

# Lists and matrices for plotting and computational purposes
N = list(range(n))
K = np.random.rand(k, n)  # Stores the localization lengths
e = np.random.rand(k, n)  # Stores the energy levels
U = np.random.rand(m.size)

for w in range(m.size):
    # Trial Loop
    for h in range(k):
        # Creating a Hamiltonian with varying disorder potential
        H = np.random.rand(n, n)
        for i in range(n):
            for j in range(n):
                if j == i:
                    H[i][j] = 2*a+m[w]*rng.random()
                elif j == i+1:
                    H[i][j] = -a
                elif j == i-1:
                    H[i][j] = -a
                else:
                    H[i][j] = 0
        H[0][n-1] = -a
        H[n-1][0] = -a
        H = H.astype(complex)

        # Finding the eigenvalues and eigenvectors of the Hamiltonian
        E, v = eig(H)
        e[h] = abs(E)
        V = np.transpose(v)

        # Finding the localization lengths
        # This is done by treating each eigenfunction as exponentially decreasing
        P = np.random.rand(n, n)
        Q = np.random.rand(n, n)
        L = np.random.rand(n)
        # Computing the probability density of each eigenfunction
        for i in range(n):
            P[i] = abs(V[i])**2
            P[i].sort()
            # Plotting the logarithm of the sorted probability density
            for j in range(n):
                Q[i, j] = -math.log(P[i, j])
            # Finding the slope of the linear plot from above
            s, y_ic, r, p, std_err = stats.linregress(N[:100], Q[i][-100:])
            # Computing the localization length from the gradient above
            L[i] = -1/(2*s)
        K[h] = L

    # Avering all localization lengths within a certain energy band
    M = abs(E.max())+1
    t = np.arange(0, M, 0.01)     # Length of energy bands
    T = np.zeros(t.size)
    # Averaging loop
    for j in range(t.size):
        z = 0  # Division constant - Measures the number of elements which is averaged over
        for h in range(k):
            for i in range(n):
                # For each element within the energy band, the corresponding localization
                # length is added to the total
                if t[j] <= e[h, i] < t[j+1]:
                    T[j] = T[j] + K[h, i]
                    z = z+1
        # Dividing the total by the number of elements
        if z != 0:
            T[j] = T[j]/z

    # Plotting the average localization lengths over each energy band
    U[w] = T.max()

R.plot(m, U)
plt.title("n="+str(n)+" k="+str(k))
plt.xlabel('m')
plt.ylabel('Loc_Leng Peak')
plt.legend()
plt.show()
