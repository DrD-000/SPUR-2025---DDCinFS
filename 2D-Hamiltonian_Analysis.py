# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

rng = np.random.default_rng()
fig, R = plt.subplots()

a = 1
n = 50
m = np.arange(0, 7, 2)

N = list(range(n**2))
# Constructing the Hamiltonian of the system
for w in range(m.size):
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

    # Finding and plotting the eigenvalues of the Hamiltonian    
    E, v = eig(H)
    E.sort()
    R.plot(N,E, label='m='+str(m[w]))
    print(str(m[w])+"done")

plt.xlabel('Eigenstate, N')
plt.ylabel('Energy, E')
plt.title("n="+str(n)+" m="+str(m))
plt.show()
   