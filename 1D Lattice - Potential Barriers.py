# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 11:56:09 2025

@author: Admin
"""

import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
rng=np.random.default_rng()
fig, R=plt.subplots()

a=1                     # - Hopping Constant
n=200                   # - #Sites
m=0                     # - Disorder Coefficient
d=0.3                   # - Wave Packet: Width
o=70                    # - Wave Packet: Origin
f=0.1                   # - Potential constant
t=np.arange(10,61,10)   # - Time intervals

# List for plotting purposes
N=list(range(n))

# Creating potential barrier(s)
F=np.random.rand(n)
for i in range(n):
    if i>100:
        F[i]=f
    else:
        F[i]=0
R.plot(N,F)

# Creating the Hamiltonian of the system with a potential given by the above function
H=np.random.rand(n,n)
for i in range(n):
    for j in range (n):
        if j==i:
            H[i][j]=2*a+F[i]
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

# Creating an initial wave function with a Gausian distribution
p=np.random.rand(n)
for i in range(n):
    p[i]=np.sqrt(d)*(np.e**(-(d**2)*((i-o)**2)))
p=p.astype(complex)

# Soling a set of simultaneous equations to find the linear coeffients of the eigenfunctions
c=np.linalg.solve(v,p)
C=np.reshape(c,(n,))

# Plotting the probability density of the wavefunction at different time intervals 
for k in t:
    # Time-dependent evolution of the individual eigenfunctions
    q=np.random.rand(n,n)
    q=q.astype(complex) 
    for i in range(n):
        for j in range (n):
            q[i,j]=C[i]*(np.exp(-E[i]*1j*k))*V[i,j]
    q=np.transpose(q)
    # Time-dependent wavefunction
    Q=np.random.rand(n)
    Q=Q.astype(complex)
    for i in range(n):
        Q[i]=np.sum(q[i])
    # Probability density
    Prob=abs(Q)**2
    
    R.plot(N,Prob,label="t="+str(k))
# Labeling
plt.title("n="+str(n)+" m="+str(m))
plt.legend()
plt.show()
