# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 14:09:20 2025

@author: Admin
"""
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
rng=np.random.default_rng()
fig, R=plt.subplots()
import math
from scipy import stats

a=1                     # - HoppingConstant
n=500                   # - #Sites
m=0.2                   # - Disorder Coefficient
k=100                   # - #Trials

# Lists and matrices for future plotting and computational purposes
N=list(range(n))
K=np.random.rand(k,n) # Stores the localization lengths of different trials
e=np.random.rand(k,n) # Stores the energy levels of different trials

# Trial loop
for l in range(k):
    # Creating a Hamiltonian with a differing disorder potential
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
    e[l]=abs(E)
    V=np.transpose(v)
    
    # Finding the localization lengths
    # This is done using the Inverse Participation Ratio
    P=np.random.rand(n,n)    
    IPR=np.random.rand(n)
    # The sum of the squares of the proability densities gives the IPR 
    for i in range(n):
        P[i]=abs(V[i])**4
        IPR[i]=np.sum(P[i])
    K[l]=IPR

# Averaging the IPRs of eigenfunctions within a certain energy band 
M=abs(E.max())+1
t=np.arange(0,M,0.01)  # Length of energy bands
T=np.zeros(t.size)
# Averaging Loop
for j in range(t.size):
    z=0  # Division Constant - Counts the number of elements averaged over
    for l in range(k):
        for i in range(n):
            # For each element within the energy band, the corresponding IPR is added to the total
            if t[j]<=e[l,i]<t[j+1]:
                T[j]=T[j]+K[l,i]
                z=z+1
    # Dividing the total by the number of elements
    if z!=0: 
        T[j]=T[j]/z

# Plotting the average IPR over each energy band
R.plot(t,T)
plt.title("n="+str(n)+" m="+str(m)+" k="+str(k))
plt.show()

