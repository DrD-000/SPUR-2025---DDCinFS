# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 09:41:31 2025

@author: Admin
"""

import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
rng=np.random.default_rng()
fig, R=plt.subplots()

a=1                     # - Hopping Constant
n=200                   # - #Sites
m=1.0                   # - Disorder Coefficient

# Creating the Hamiltonian matrix, and a list N for plotting purposes
N=list(range(n))
H=np.random.rand(n,n)

# Replacing each value of the Hamiltonian with the given theoretical value
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

# Finding the eigenvalues of the system, and ordering them in an increasing list
E,v=eig(H)
E.sort()

# Plotting the energy levels of the system
R.plot(N,E)
plt.show()







