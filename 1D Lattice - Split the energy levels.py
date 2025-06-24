# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 09:41:02 2025

@author: Admin
"""

import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
rng=np.random.default_rng()
fig, R=plt.subplots()

a=1                     # - Hopping Constant
b=2                     # - Hopping Constant
n=200                   # - #Sites
m=0.0                   # - Disorder Coefficient

# List for plotting purposes
N=list(range(n))

# Creating a Hamiltonian with periodically changing hopping constants
# Split into even and odd sites 
H=np.random.rand(n,n)
for i in range(n):
    if i-i*(i%2)==0:
        for j in range (n):
            if j==i:
                H[i][j]=2/(a*b)+m*rng.random()
            elif j==i+1:
                H[i][j]=-2/(a*(a+b))
            elif j==i-1:
                H[i][j]=-2/(b*(a+b))
            else:
                H[i][j]=0
    elif i-i*(i%2)==i:
        for j in range(n):
            if j==i:
                H[i][j]=2/(a*b)+m*rng.random()
            elif j==i+1:
                H[i][j]=-2/(b*(a+b))
            elif j==i-1:
                H[i][j]=-2/(a*(a+b))
            else:
                H[i][j]=0
        
H[0][n-1]=-2/(b*(a+b))
H[n-1][0]=H[n-2][n-3]
H=H.astype(complex)

# Finding eigenvalues of the Hamiltonian
E,v=eig(H)
E.sort()

# Ploting the energy levels of the system
R.plot(N,E)
plt.show()
