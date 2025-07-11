#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 11:19:28 2025

@author: danielWH
"""
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
import math
from scipy import stats

rng = np.random.default_rng()
fig, R = plt.subplots()


a = 1
n = 20
m = np.arange(1, 6, 1)
o_x = 10
o_y = 10
t = np.arange(1,202,5)   
K = 100                 # - #Trials

N = list(range(n**2))

for w in range(m.size):
    L = np.random.rand(K,t.size)
    for h in range(K):
        H = np.random.rand(n**2,n**2)

        for i in range(n**2):
            for j in range(n**2):
                if j == i:
                    H[i,j] = 4*a+m[w]*rng.random()
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

        P = np.random.rand(t.size,n**2)
        S = np.random.rand(t.size)
        r = np.random.rand(t.size, n**2)

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

            # Storing probability density functions of each time interval
            for i in range(n**2):
                P[k,i] = abs(Q[i])**2
                P[k].sort()
        
            for i in range(n**2):
                r[k,i] = -math.log(P[k,i])
            s, y_ic, reg, p, std_err = stats.linregress(N[:int(n**2/5)], r[k][int(-(n**2)/5):])
            S[k] = -1/(4*s)
            L[h,k] = S[k]
        print('('+str(m[w])+', '+str(h)+') done')
    L_avg = np.random.rand(t.size)
    for k in range(t.size):
        L_avg[k] = np.sum(np.transpose(L)[k])/K

    R.plot(t, L_avg, label="m="+str(m[w]))
    
plt.title("n="+str(n)+" K="+str(K)+" o_x="+str(o_x)+" o_y="+str(o_y))
plt.xlabel('Time, t')
plt.ylabel('Localization Length, l')
plt.legend()
plt.show()