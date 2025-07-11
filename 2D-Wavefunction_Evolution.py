#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 10:19:17 2025

@author: danielWH
"""

import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')
rng = np.random.default_rng()

a = 1
n = 50
m = 0.0
o_x = 0
o_y = 0
t = np.arange(5,21,5)   
K = 1                 # - #Trials

N = list(range(n**2))
G = np.random.rand(n,n)
for i in range(n):
    G[i] = list(range(n))
M = np.random.rand(n,n)
for i in range(n):
    for j in range(n):
        M[i,j] = i

s = np.random.rand(K,t.size,n,n)
for h in range(K):
    H = np.random.rand(n**2,n**2)

    for i in range(n**2):
        for j in range(n**2):
            if j == i:
                H[i,j] = 4*a+m*rng.random()
            elif j == i+1 or j == i-1:
                H[i,j] = -a
            elif j == i+n or j == i-n:
                H[i,j] = -a
            else:
                H[i,j] = 0

    for i in range(n):
        if i*n-1 in range(n**2):
            H[i*n, i*n-1] = 0
        if (i+1)*n in range(n**2):
            H[(i+1)*n-1,(i+1)*n] = 0

    for i in range(n):
        H[i, i+n*(n-1)] = -a
        H[i+n*(n-1), i] = -a
        H[i*n, (i+1)*n-1] = -a
        H[(i+1)*n-1,i*n] = -a
    
    E, v = eig(H)
    V = v.transpose()

    p=np.zeros(n**2)
    p[o_y*n+o_x] = 1
    p=p.astype(complex)
    
    # Finding the coefficients of the linear superposition of eigenfunctions
    c=np.linalg.solve(v,p)

    P = np.random.rand(t.size,n,n)
    C = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for k in range(t.size):
        
        # Individual eigenfunctions as functions of time
        q=np.random.rand(n**2,n**2)
        q=q.astype(complex)
        for i in range(n**2):
            for j in range (n**2):
                q[i,j]=c[i]*(np.exp(-E[i]*1j*t[k]))*V[i,j]
        q=np.transpose(q)    
        
        # Linear superposition of time-dependent eigenfundtions
        Q=np.random.rand(n**2)
        Q=Q.astype(complex)
        for i in range(n**2):
            Q[i]=np.sum(q[i])

        R=np.random.rand(n,n)
        R=R.astype(complex)
        for i in range(n):
            R[i]=Q[i*n:(i+1)*n]
        
        # Storing probability density functions of each time interval
        for i in range(n):
            for j in range(n):
                P[k,i,j] = abs(R[i,j])**2
        s[h,k] = P[k]

S = np.zeros((t.size,n,n))
for h in range(K):
    for k in range(t.size):
        for i in range(n):
            for j in range(n):
                S[k,i,j] = S[k,i,j]+s[h,k,i,j]
for k in range(t.size):
    S[k,i,j] = S[k,i,j]/K
for k in range(t.size):
    ax.plot_surface(M, G, S[k], color=C[k], alpha=0.3, label="t="+str(t[k]))

ax.set_title('n='+str(n)+' m='+str(m)+' K='+str(K)+' o_x='+str(o_x)+' o_y='+str(o_y))
ax.set_zlabel('Probability density, P')
plt.legend()
plt.show()
