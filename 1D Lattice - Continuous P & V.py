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

a=1                     # - Constant
n=200                   # - #Sites
m=0                     # - Disorder Coefficient
d=0.3                   # - Wave Packet: Width
o=70                    # - Wave Packet: Origin
f=0.1                   # - Potential constant
t=np.arange(10,61,10)   # - Time intervals

N=list(range(n))

F=np.random.rand(n)
for i in range(n):
    if i>100:
        F[i]=f
    else:
        F[i]=0
R.plot(N,F)

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

E,v=eig(H)
V=np.transpose(v)
    
p=np.random.rand(n)
for i in range(n):
    p[i]=np.sqrt(d)*(np.e**(-(d**2)*((i-o)**2)))
p=p.astype(complex)

c=np.linalg.solve(v,p)
C=np.reshape(c,(n,))

for k in t:
    
    q=np.random.rand(n,n)
    q=q.astype(complex)

    for i in range(n):
        for j in range (n):
            q[i,j]=C[i]*(np.exp(-E[i]*1j*k))*V[i,j]
    q=np.transpose(q)

    Q=np.random.rand(n)
    Q=Q.astype(complex)

    for i in range(n):
        Q[i]=np.sum(q[i])

    Prob=abs(Q)**2
    
    R.plot(N,Prob,label="t="+str(k))
plt.title("n="+str(n)+" m="+str(m))
plt.legend()
plt.show()
