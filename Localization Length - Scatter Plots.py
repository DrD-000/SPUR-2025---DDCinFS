# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 09:17:41 2025

@author: Admin
"""

import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
rng=np.random.default_rng()
fig, R=plt.subplots()
import math
from scipy import stats

a=1                     # - Hopping Constant
n=500                   # - #Sites
m=10.0                  # - Disorder Coefficient
k=5                     # - #Trials

# Lists and matrices for plotting and computational purposes
N=list(range(n))
K=np.random.rand(k,n)
s=np.random.rand(k,n)

# Trial Loop
for l in range(k):
    # Creating a Hamiltonian of the system, with a random potential (disorder)
    H=np.random.rand(n,n)
    for i in range(n):
        for j in range (n):
            if j==i:
                H[i][j]=2*a+m*rng.random()
            elif j==i+1:
                H[i][j]=-a
            elif j==i-1:
                H[i][j]=-a
            else:
                H[i][j]=0
    H[0][n-1]=-a
    H[n-1][0]=-a
    H=H.astype(complex)

    # Finding the eigenvalues and eigenvectors of the Hamiltonian
    E,v=eig(H)
    V=np.transpose(v)

    # Finding and plotting the localization length of each eigenstate
    # This is done by treating each wavefunction as exponentially decreasing 
    P=np.random.rand(n,n)
    Q=np.random.rand(n,n)
    L=np.random.rand(n)
    # Computing probability density of each wavefunction, and sorting it increasingly
    for i in range(n):
        P[i]=abs(V[i])**2
        P[i].sort()
        # Computing the logarithm of each probability density
        for j in range(n):
            Q[i,j]=-math.log(P[i,j])
        # Finding the slope of the line made by the logarithm of the probability density
        s,y_ic,r,p,std_err = stats.linregress(N,Q[i])
        # Computing the localization lengths from the gradients
        L[i]=-1/(2*s)
    # Scatter plot of corresponding localization lengths and energy levels
    plt.scatter(E,L,s=2)
# Labeling plot
plt.title("n="+str(n)+" m="+str(m)+" k="+str(k))
plt.legend()
plt.show()


    
    
    
    
    