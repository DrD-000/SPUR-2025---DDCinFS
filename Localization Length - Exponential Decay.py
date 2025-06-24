# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 10:58:08 2025

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
m=2.0                   # - Disorder Coefficient
k=100                   # - #Trials

# Lists and matrices for plotting and computational purposes
N=list(range(n))
K=np.random.rand(k,n) # Stores the localization lengths
e=np.random.rand(k,n) # Stores the energy levels

# Trial Loop
for l in range(k):
    # Creating a Hamiltonian with varying disorder potential
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
    # This is done by treating each eigenfunction as exponentially decreasing
    P=np.random.rand(n,n)
    Q=np.random.rand(n,n)
    L=np.random.rand(n)
    # Computing the probability density of each eigenfunction
    for i in range(n):
        P[i]=abs(V[i])**2
        P[i].sort()
        # Plotting the logarithm of the sorted probability density
        for j in range(n):
            Q[i,j]=-math.log(P[i,j])
        # Finding the slope of the linear plot from above
        s,y_ic,r,p,std_err = stats.linregress(N[:100],Q[i][-100:])
        # Computing the localization length from the gradient above
        L[i]=-1/(2*s)
    K[l]=L

# Avering all localization lengths within a certain energy band
M=abs(E.max())+1         
t=np.arange(0,M,0.01)     # Length of energy bands
T=np.zeros(t.size)
# Averaging loop
for j in range(t.size):
    z=0 # Division constant - Measures the number of elements which is averaged over
    for l in range(k):
        for i in range(n):
            # For each element within the energy band, the corresponding localization
            # length is added to the total
            if t[j]<=e[l,i]<t[j+1]:
                T[j]=T[j]+K[l,i]
                z=z+1
    # Dividing the total by the number of elements
    if z!=0:
        T[j]=T[j]/z
 
# Plotting the average localization lengths over each energy band           
R.plot(t,T)
plt.title("n="+str(n)+" m="+str(m)+" k="+str(k))
plt.show()

