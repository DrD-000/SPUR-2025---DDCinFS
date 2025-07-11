#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 09:20:17 2025

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
n = np.arange(5, 20, 1)       # - #Sites
m = np.arange(2, 11, 2)     # - Disorder Coefficient
t = 1000                         # - Time intervals
K = 1000                            # - #Trials

# List for plotting purposes
g_s = np.random.rand(m.size, n.size)
L_nm = np.random.rand(m.size, n.size)
print("m:"+str(m.size)+" n:"+str(n.size))
for w in range(m.size):
    for z in range(n.size): 
        o_x = int(n[z]/2)
        o_y = int(n[z]/2)
        N = list(range(n[z]**2))
        L = np.random.rand(K)
        # Trial loop
        for h in range(K):
            H = np.random.rand(n[z]**2,n[z]**2)

            for i in range(n[z]**2):
                for j in range(n[z]**2):
                    if j == i:
                        H[i,j] = 4*a+m[w]*rng.random()
                    elif j == i+1 or j == i-1:
                        H[i,j] = -a
                    elif j == i+n[z] or j == i-n[z]:
                        H[i,j] = -a
                    else:
                        H[i,j] = 0

            for i in range(n[z]):
                if i*n[z]-1 in range(n[z]**2):
                    H[i*n[z], i*n[z]-1] = 0
                if (i+1)*n[z] in range(n[z]**2):
                    H[(i+1)*n[z]-1,(i+1)*n[z]] = 0

            for i in range(n[z]):
                H[i, i+n[z]*(n[z]-1)] = -a
                H[i+n[z]*(n[z]-1), i] = -a
                H[i*n[z], (i+1)*n[z]-1] = -a
                H[(i+1)*n[z]-1,i*n[z]] = -a
        
            E, v = eig(H)
            V = v.transpose()

            p=np.zeros(n[z]**2)
            p[o_y*n[z]+o_x] = 1
            p=p.astype(complex)
        
            # Finding the coefficients of the linear superposition of eigenfunctions
            c=np.linalg.solve(v,p)

            P = np.random.rand(n[z]**2)
            
            # Individual eigenfunctions as functions of time
            q=np.random.rand(n[z]**2,n[z]**2)
            q=q.astype(complex)
            for i in range(n[z]**2):
                for j in range (n[z]**2):
                    q[i,j]=c[i]*(np.exp(-E[i]*1j*t))*V[i,j]
            q=np.transpose(q)    
            
            # Linear superposition of time-dependent eigenfundtions
            Q=np.random.rand(n[z]**2)
            Q=Q.astype(complex)
            for i in range(n[z]**2):
                Q[i]=np.sum(q[i])

            # Storing probability density functions of each time interval
            for i in range(n[z]**2):
                P[i] = abs(Q[i])**2
                P.sort()
            
            r = np.random.rand(n[z]**2)
            for i in range(n[z]**2):
                r[i] = -math.log(P[i])
            s, y_ic, reg, pr, std_err = stats.linregress(N[:int(n[z]**2/5)], r[int(-(n[z]**2)/5):])
            S = -1/(4*s)
            L[h] = S

        L_nm[w,z] = np.sum(L)/K
        g_s[w,z] = np.exp((-(n[z]**2)/L_nm[w,z]))
        print("{"+str(m[w])+", "+str(n[z])+"} done")

ln_gs = np.random.rand(m.size, n.size)
ln_ns = np.random.rand(m.size, n.size)
for w in range(m.size):
    for z in range(n.size):
        ln_gs[w, z] = math.log(g_s[w, z])
        ln_ns[w, z] = math.log(n[z]**2)

U_s = np.arange(0, n.size+1, 3)
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
    g_w[z] = n[z]**0
    ln_gw[z] = math.log(g_w[z])
    ln_nw[z] = math.log(n[z])

U_w = np.arange(0, n.size+1, 3)
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
