# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 11:34:05 2025

@author: Admin
"""

import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
rng=np.random.default_rng()
fig, R=plt.subplots()

a=1                     # - Hopping Constant
n=200                   # - #Sites
m=0.5                   # - Disorder Coefficient
d=0.5                   # - Wave Packet: Width
o=100                   # - Wave Packet: Origin
u=0                     # - Wave Packet: Momentum
t=np.arange(10,51,10)   # - Time intervals
K=1                     # - #Trials

# List for plotting purposes
N=list(range(n))

# Tensor for averaging the wavefunctions of different trials
P=np.random.rand(K,t.size,n)

# Trial loop
for l in range(K):
    
    # Creating the Hamiltonian of the system, with corresponding theoretical values
    H=np.random.rand(n,n)
    
    for i in range(n):
        for j in range (n):
            if j==i:
                if i<201:
                    H[i][j]=2*a+m*rng.random()
                else: 
                    H[i][j]=2*a
            elif j==i+1:
                H[i][j]=-a
            elif j==i-1:
                H[i][j]=-a
            else:
                H[i][j]=0
    H[0][n-1]=-a
    H[n-1][0]=-a
    H=H.astype(complex)

    # Finding the eigenvalues and eigenvectors of the system
    E,v=eig(H)
    V=np.transpose(v)  #Transpose gives the eigenvectors horizontally
    
    # Creating an inital wavefunction at time t=0, with a Gaussian distribution
    p=np.random.rand(n)
    p=p.astype(complex)
    for i in range(n):
        p[i]=np.sqrt(d)*(np.exp(1j*u*(i-o)))*(np.exp(-(d**2)*((i-o)**2)))
    p=p.astype(complex)
    
    # Finding the coefficients of the linear superposition of eigenfunctions
    c=np.linalg.solve(v,p)
    C=np.reshape(c,(n,))

    # Computing the wave function at different time intervals
    for k in range(t.size):
        
        # Individual eigenfunctions as functions of time
        q=np.random.rand(n,n)
        q=q.astype(complex)
        for i in range(n):
            for j in range (n):
                q[i,j]=C[i]*(np.exp(-E[i]*1j*t[k]))*V[i,j]
        q=np.transpose(q)

        # Linear superposition of time-dependent eigenfundtions
        Q=np.random.rand(n)
        Q=Q.astype(complex)
        for i in range(n):
            Q[i]=np.sum(q[i])

        # Storing probability density functions of each time interval
        P[l,k]=abs(Q)**2

# Averaging the different probability densities of the same interval
L=np.zeros((t.size,n))
for k in range(K):
    for i in range(t.size):
        for j in range(n):
            L[i,j]=L[i,j]+P[k,i,j]
for i in range(t.size):
    for j in range(n):
        L[i,j]=L[i,j]/K
# Plotting average probability density of each time interval
for i in range (t.size):
    R.plot(N,L[i],label="t="+str(t[i]))    
    
# Labeling plot
plt.title("n="+str(n)+" m="+str(m)+" K="+str(K)+" d="+str(d)+" u="+str(u)+" o="+str(o))
plt.legend()
plt.show()

