#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 09:33:43 2025

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
m = np.arange(0.5, 20.1, 0.5)
K = 100

N = list(range(n**2))
L = np.random.rand(K, n**2)
e = np.random.rand(K, n**2)
L_avg = np.random.rand(m.size)
# Constructing the Hamiltonian of the system
H = np.zeros((n**2,n**2))
for i in range(n**2):
    for j in range(n**2):
        if j == i:
            H[i,j] = 4*a
        elif j == i+1 or j == i-1:
            H[i,j] = -a
        elif j == i+n or j == i-n:
            H[i,j] = -a

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

for w in range(m.size):
    for h in range(K):
        for i in range(n**2):
            for j in range(n**2):
                if j == i:
                    H[i,j] = 4*a+m[w]*rng.random()

        # Finding and plotting the eigenvalues of the Hamiltonian    
        E, v = eig(H)
        e[h] = abs(E)
        V = np.transpose(v)
    
        # Finding the localization lengths
        # This is done by treating each eigenfunction as exponentially decreasing
        P=np.random.rand(n**2, n**2)
        Q=np.random.rand(n**2, n**2)
        l=np.random.rand(n**2)
        # Computing the probability density of each eigenfunction
        for i in range(n**2):
            P[i]=abs(V[i])**2
            P[i].sort()
            # Plotting the logarithm of the sorted probability density
            for j in range(n**2):
                Q[i,j]=-math.log(P[i,j])
            # Finding the slope of the linear plot from above
            s,y_ic,r,p,std_err = stats.linregress(N[:int((n**2)/5)],Q[i][int(-(n**2)/5):])
            # Computing the localization length from the gradient above
            l[i]=-1/(4*s)
        L[h]=l
        print('('+str(m[w])+', '+str(h)+") done")

    # Avering all localization lengths within a certain energy band
    Max=abs(E.max())+1         
    U=np.arange(0,Max,0.05)     # Length of energy bands
    S=np.zeros(U.size)
    # Averaging loop
    for u in range(U.size):
        z=0 # Division constant - Measures the number of elements which is averaged over
        for h in range(K):
            for i in range(n**2):
                # For each element within the energy band, the corresponding localization
                # length is added to the total
                if u+1 in range(U.size) and U[u]<=e[h,i]<U[u+1]:
                    S[u]=S[u]+L[h,i]
                    z=z+1
        # Dividing the total by the number of elements
        if z!=0:
            S[u]=S[u]/z

    if S.size%2 == 1:
        L_avg[w]=np.sum(S[int(S.size/2)-2:int(S.size/2)+1])/3
    elif S.size%2 == 0:
        L_avg[w] = np.sum(S[int(S.size/2)-3:int(S.size/2)+1])/4
    else:
        print("S-error")
# Plotting the average localization lengths over each energy band           
R.plot(m,L_avg)
plt.title("n="+str(n)+" K="+str(K))
plt.xlabel('Disorder coefficient, m')
plt.ylabel('Average localization length, l')
plt.show()