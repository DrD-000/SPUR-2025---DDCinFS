#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 15:23:34 2025

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
n = np.arange(20, 101, 10)       # - #Sites
m = np.arange(0.5, 2.6, 0.5)     # - Disorder Coefficient
t = 1000                         # - Time intervals
K = 10                            # - #Trials

# List for plotting purposes
g_s = np.random.rand(m.size, n.size)
L_nm = np.random.rand(m.size, n.size)
print("m:"+str(m.size)+" n:"+str(n.size))
for w in range(m.size):
    for z in range(n.size): 
        o = int(n[z]/2)                      # - Wave Packet: Origin
        N = list(range(n[z]))
        L = np.random.rand(K)
        # Trial loop
        for h in range(K):

            # Creating the Hamiltonian of the system, with corresponding theoretical values
            H = np.random.rand(n[z], n[z])

            for i in range(n[z]):
                for j in range(n[z]):
                    if j == i:
                        if i < n[z]:
                            H[i][j] = 2*a+m[w]*rng.random()
                        else:
                            H[i][j] = 2*a
                    elif j == i+1:
                        H[i][j] = -a
                    elif j == i-1:
                        H[i][j] = -a
                    else:
                        H[i][j] = 0
            H[0][n[z]-1] = -a
            H[n[z]-1][0] = -a
            H = H.astype(complex)

            # Finding the eigenvalues and eigenvectors of the system
            E, v = eig(H)
            V = np.transpose(v)  # Transpose gives the eigenvectors horizontally

            # Creating an inital wavefunction at time t=0, with a Gaussian distribution
            p = np.zeros(n[z])
            p[o] = 1
            p = p.astype(complex)

            # Finding the coefficients of the linear superposition of eigenfunctions
            c = np.linalg.solve(v, p)
            C = np.reshape(c, (n[z],))

            # Matrix for averaging the wavefunctions of different trials
            P = np.random.rand(n[z])
            r = np.random.rand(n[z])
            # Computing the wave function at different time intervals

            # Individual eigenfunctions as functions of time
            q = np.random.rand(n[z], n[z])
            q = q.astype(complex)
            for i in range(n[z]):
                for j in range(n[z]):
                    q[i, j] = C[i]*(np.exp(-E[i]*1j*t))*V[i, j]
            q = np.transpose(q)

            # Linear superposition of time-dependent eigenfundtions
            Q = np.random.rand(n[z])
            Q = Q.astype(complex)
            for i in range(n[z]):
                Q[i] = np.sum(q[i])

            # Storing probability density functions of each time interval
            P = abs(Q)**2
            P.sort()
            for i in range(n[z]):
                if P[i] >= 1e-50:
                    r[i] = -math.log(P[i])
            s, y_ic, reg, pr, std_err = stats.linregress(N[:int(n[z]/5)], r[int(-n[z]/5):])
            S = -1/(2*s)
            L[h] = S

        L_nm[w,z] = np.sum(L)/K
        g_s[w,z] = np.exp((-n[z]/L_nm[w,z]))
        print("{"+str(m[w])+", "+str(n[z])+"} done")

ln_gs = np.random.rand(m.size, n.size)
ln_ns = np.random.rand(m.size, n.size)
for w in range(m.size):
    for z in range(n.size):
        ln_gs[w, z] = math.log(g_s[w, z])
        ln_ns[w, z] = math.log(n[z])

U_s = np.arange(0, n.size+1, 5)
B_s = np.random.rand(m.size, U_s.size-1)
G_s = np.random.rand(m.size, U_s.size-1)
for w in range(m.size):
    for u in range(U_s.size):
        if u+1 in range(U_s.size):
            T_s, y_ic_s, reg_s, pr_s, std_err_s = stats.linregress(ln_ns[w, U_s[u]:U_s[u+1]], ln_gs[w, U_s[u]:U_s[u+1]])
            B_s[w, u] = T_s
            G_s[w, u] = (np.sum(ln_gs[w, U_s[u]:U_s[u+1]]))/(ln_gs[w, U_s[u]:U_s[u+1]].size)
    R.scatter(G_s[w], B_s[w], label='m='+str(m[w]))

g_w = np.random.rand(n.size)
ln_gw = np.random.rand(n.size)
ln_nw = np.random.rand(n.size)
for z in range(n.size):
    g_w[z] = 1/n[z]
    ln_gw[z] = math.log(g_w[z])
    ln_nw[z] = math.log(n[z])

U_w = np.arange(0, n.size+1, 5)
B_w = np.random.rand(U_w.size-1)
G_w = np.random.rand(U_w.size-1)
for u in range(U_w.size):
    if u+1 in range(U_s.size):
        T_w, y_ic_w, reg_w, pr_w, std_err_w = stats.linregress(ln_nw[U_w[u]:U_w[u+1]], ln_gw[U_w[u]:U_w[u+1]])
        B_w[u] = T_w
        G_w[u] = (np.sum(ln_gw[U_w[u]:U_w[u+1]]))/(ln_gw[U_w[u]:U_w[u+1]].size)
R.scatter(G_w, B_w, color='k', marker='X')

# Labeling plot
plt.title("K="+str(K)+" t="+str(t))
plt.xlabel('Logarithmic Conductance, ln(g)')
plt.ylabel('Scaling function, B(g)')
plt.legend()
plt.show()
