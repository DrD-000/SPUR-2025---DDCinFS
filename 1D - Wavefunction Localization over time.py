#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 10:29:17 2025

@author: danielWH
"""

import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
import math
from scipy import stats
rng = np.random.default_rng()
fig, R = plt.subplots()

a = 1                         # - Hopping Constant
n = 500                       # - #Sites
m = np.arange(0.2, 1.3, 0.2)  # - Disorder Coefficient
d = 0.5                       # - Wave Packet: Width
o = 100                       # - Wave Packet: Origin
M = 0                         # - Wave Packet: Momentum
t = np.arange(0, 51, 1.0)     # - Time intervals
K = 100                       # - #Trials

# List for plotting purposes
N = list(range(n))

for w in range(m.size):
    L = np.random.rand(K, t.size)
    # Trial loop
    for h in range(K):

        # Creating the Hamiltonian of the system, with corresponding theoretical values
        H = np.random.rand(n, n)

        for i in range(n):
            for j in range(n):
                if j == i:
                    if i < 201:
                        H[i][j] = 2*a+m[w]*rng.random()
                    else:
                        H[i][j] = 2*a
                elif j == i+1:
                    H[i][j] = -a
                elif j == i-1:
                    H[i][j] = -a
                else:
                    H[i][j] = 0
        H[0][n-1] = -a
        H[n-1][0] = -a
        H = H.astype(complex)

        # Finding the eigenvalues and eigenvectors of the system
        E, v = eig(H)
        V = np.transpose(v)  # Transpose gives the eigenvectors horizontally

        # Creating an inital wavefunction at time t=0, with a Gaussian distribution
        p = np.random.rand(n)
        p = p.astype(complex)
        for i in range(n):
            p[i] = np.sqrt(d)*(np.exp(1j*M*(i-o)))*(np.exp(-(d**2)*((i-o)**2)))
        p = p.astype(complex)

        # Finding the coefficients of the linear superposition of eigenfunctions
        c = np.linalg.solve(v, p)
        C = np.reshape(c, (n,))

        # Matrix for averaging the wavefunctions of different trials
        P = np.random.rand(t.size, n)
        r = np.random.rand(t.size, n)
        S = np.random.rand(t.size)
        # Computing the wave function at different time intervals
        for k in range(t.size):

            # Individual eigenfunctions as functions of time
            q = np.random.rand(n, n)
            q = q.astype(complex)
            for i in range(n):
                for j in range(n):
                    q[i, j] = C[i]*(np.exp(-E[i]*1j*t[k]))*V[i, j]
            q = np.transpose(q)

            # Linear superposition of time-dependent eigenfundtions
            Q = np.random.rand(n)
            Q = Q.astype(complex)
            for i in range(n):
                Q[i] = np.sum(q[i])

            # Storing probability density functions of each time interval
            P[k] = abs(Q)**2
            P[k].sort()
            for i in range(n):
                if P[k, i] >= 1e-50:
                    r[k, i] = -math.log(P[k, i])
            s, y_ic, reg, p, std_err = stats.linregress(N[:100], r[k][-100:])
            S[k] = -1/(2*s)
        L[h] = S

    L_avg = np.random.rand(t.size)
    for k in range(t.size):
        L_avg[k] = np.sum(np.transpose(L)[k])/K

    R.plot(t, L_avg, label='m='+str(m[w]))
# Labeling plot
plt.title("n="+str(n)+" K="+str(K)+" d="+str(d)+" M="+str(M)+" o="+str(o))
plt.xlabel('Time, t')
plt.ylabel('Loc_Leng')
plt.legend()
plt.show()
